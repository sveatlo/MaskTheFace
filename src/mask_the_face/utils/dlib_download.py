import tempfile
import shutil
import bz2

import requests
from tqdm import tqdm


dlib_model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

def download_dlib_model(dest="dlib_models/shape_predictor_68_face_landmarks.dat"):
    print(f"Downloading {dlib_model_url}...")

    decompressor = bz2.BZ2Decompressor()
    req = requests.get(dlib_model_url, stream=True)

    total_size_in_bytes= int(req.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(dest, 'wb') as fp:
        for chunk in req.iter_content():
            progress_bar.update(len(chunk))
            decompressed_data = decompressor.decompress(chunk)
            fp.write(decompressed_data)
