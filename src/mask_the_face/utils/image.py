import cv2
import numpy as np


def match_brightness(img1, img2):
    img1_brightness = get_avg_brightness(img1)
    img2_brightness = get_avg_brightness(img2)
    delta = 1 + (img1_brightness - img2_brightness) / 255
    return change_brightness(img2, delta)

def match_saturation(img1, img2):
    img1_saturation = get_avg_saturation(img1)
    img2_saturation = get_avg_saturation(img2)
    delta = 1 - (img1_saturation - img2_saturation) / 255
    return change_saturation(img2, delta)






def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(s)

def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(img_hsv)
    v = v*value
    v = np.clip(v,0,255)
    img_hsv = cv2.merge([h,s,v])
    return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(img_hsv)
    s = s*value
    s = np.clip(s,0,255)
    img_hsv = cv2.merge([h,s,v])
    return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

