#!/usr/bin/env python

import os
import sh
import random
import cv2
import numpy as np

def resize_by_short_edge(img, size):
    h, w = img.shape[0], img.shape[1]
    if h < w:
        scale = w / float(h)
        new_width = int(size * scale)
        img = cv2.resize(img, (new_width, size))
    else:
        scale = h / float(w)
        new_height = int(size * scale)
        img = cv2.resize(img, (size, new_height))
    return img


def shuffle(df, n=1, axis=0):
   # df = df.copy()
   # for _ in range(n):
   #     df.apply(np.random.shuffle, axis=axis)
   # return df
    random.shuffle(df)
    return df

def burst_frames_to_shm(vid_path, shm_dir_path):
    """
    - To burst frames in a temporary directory in shared memory.
    - Directory name is chosen as random 128 bits so as to avoid clash during
      parallelization
    - Returns path to directory containing frames for the specific video
    """
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    target_mask = os.path.join(temp_dir, '%04d.jpg')
    try:
        sh.ffmpeg('-i', vid_path,
                  '-q:v', str(1),
                  '-r', 8,
                  '-f', 'image2', target_mask)
    except Exception as e:
        print(repr(e))
    return temp_dir


def ensure_output_dir_exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
