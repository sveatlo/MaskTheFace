import os
import random
from configparser import ConfigParser
# 3rd party imports
import dlib
import cv2
import numpy as np
# local imports
import utils

module_dir_path = os.path.dirname(os.path.realpath(__file__))

class Masker(object):
    """
    Masker is a class used to apply masks to faces in images
    """

    def __init__(self, predictor_model_path=module_dir_path+'/dlib_models/shape_predictor_68_face_landmarks.dat'):
        super().__init__()

        if not os.path.exists(predictor_model_path):
            utils.download_dlib_model(predictor_model_path)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_model_path)

        self._masks_dir = os.path.join(module_dir_path, "masks")
        self._masks_cfg = self._read_masks_cfg(os.path.join(self._masks_dir,"masks.cfg"))
        self._angle_threshold = 13

    def apply_mask(self, image, mask_type="surgical", mask_pattern=None, mask_pattern_weight=0.5, mask_color=None, mask_color_weight=0.5):
        """TODO: Docstring for apply_mask.

        :param image: Instance of OpenCV2's Image class
        :returns: Tuple(masked_image, image_mask, face_masks, face_positions)

        """
        image_original = image.copy()

        masks = [] # separate masks for each face
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        face_locations = self.detector(image, 1)
        for (i, face_location) in enumerate(face_locations):
            shape = self.predictor(image, face_location)
            shape = utils.shape_to_np(shape)
            face_landmarks = utils.shape_to_landmarks(shape)
            face_location = utils.rect_to_bb(face_location)
            six_points_on_face, angle = utils.get_six_points(face_landmarks, image)

            masked_image, face_mask = self._mask_face(
                image,
                face_location,
                six_points_on_face,
                angle,
                mask_type=mask_type,
                mask_pattern=mask_pattern,
                mask_pattern_weight=mask_pattern_weight,
                mask_color=mask_color,
                mask_color_weight=mask_color_weight,
            )
            image = masked_image # next face on the already masked image
            masks.append(face_mask)
            #  print(mask.shape, face_mask.shape)
            mask = cv2.bitwise_or(mask, face_mask)

        return image, mask, masks, face_locations

    def _mask_face(self, image, face_location, six_points, angle, mask_type="surgical", mask_pattern=None, mask_pattern_weight=0.5, mask_color=None, mask_color_weight=0.5):
        """
        _mask_face masks single face in an image

        :param image: an instance of opencv2 image
        :param face_location: tuple (x1, x2, y2, x1) - bounding box of face
        :param angle: float number showing face rotation angle
        :param mask_type: type of mask to apply
        :returns: Tuple(masked_image, mask_binary)

        """

        mask_type_ = mask_type
        if mask_type_ == "random":
            mask_type_ = random.choice(self.available_masks())
        if mask_type_ == "empty" or mask_type_ == "inpaint":
            mask_type_ = "surgical_blue"

        mask_orientation = "front"
        if angle < -self._angle_threshold:
            mask_orientation = "right"
        elif angle > self._angle_threshold:
            mask_orientation = "left"

        w, h, _ = image.shape
        face_height = face_location[2] - face_location[0]
        face_width = face_location[1] - face_location[3]
        face_crop_img = image[face_location[0] : face_location[2], face_location[3] : face_location[1], :]

        try:
            mask_cfg = self._masks_cfg[mask_type_][mask_orientation]
        except Exception as e:
            raise Exception(f"unknown mask type + orientation combination {mask_type_}_{mask_orientation}")

        mask_image_path = os.path.join(self._masks_dir, mask_cfg['template'])
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            raise Exception(f"mask image ({mask_image_path}) not found")

        if mask_pattern is not None:
            mask_pattern_path = os.path.join(self._masks_dir, "textures", mask_pattern)
            if not os.path.exists(mask_pattern_path):
                mask_pattern_path = mask_pattern

            mask_image = utils.texture_the_mask(mask_image, mask_pattern_path, mask_pattern_weight)
        if mask_color is not None:
            mask_image = utils.color_the_mask(mask_image, mask_color, mask_color_weight)

        mask_line = np.float32([
            mask_cfg['mask_a'],
            mask_cfg['mask_b'],
            mask_cfg['mask_c'],
            mask_cfg['mask_f'],
            mask_cfg['mask_e'],
            mask_cfg['mask_d'],
        ])


        # Warp the mask
        M, _ = cv2.findHomography(mask_line, six_points)
        mask_image = cv2.warpPerspective(mask_image, M, (h, w))
        dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
        mask_binary = mask_image[:, :, 3]

        # match brightness+saturation
        #  mask_image = utils.match_brightness(face_crop_img, mask_image)
        #  mask_image = utils.match_saturation(face_crop_img, mask_image)

        # Apply mask
        mask_inv = cv2.bitwise_not(mask_binary)
        img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        img_fg = cv2.bitwise_and(mask_image, mask_image, mask=mask_binary)
        masked_img = cv2.add(img_bg, img_fg[:, :, 0:3])
        if mask_type == "empty" or mask_type == "inpaint":
            masked_img = img_bg
        if mask_type == "inpaint":
            masked_img = cv2.inpaint(masked_img, mask_binary, 3, cv2.INPAINT_TELEA)


        return masked_img, mask_binary




    #
    # utility methods
    #

    def available_masks(self):
        return list(self._masks_cfg.keys())

    def _read_masks_cfg(self, config_filename):
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(config_filename)

        cfg = {}

        for section_name, section in parser.items():
            if section_name == "DEFAULT":
                continue

            mask_type = section_name
            mask_type = mask_type.removesuffix("_left")
            mask_type = mask_type.removesuffix("_right")

            mask_orientation = "front"
            if "_left" in section_name:
                mask_orientation = "left"
            elif "_right" in section_name:
                mask_orientation = "right"

            if mask_type not in cfg:
                cfg[mask_type] = {}
            if mask_orientation not in cfg[mask_type]:
                cfg[mask_type][mask_orientation] = {}

            for key, value in parser.items(section_name):
                cfg[mask_type][mask_orientation][key] = value if key == "template" else tuple(int(s) for s in value.split(","))

        return cfg
