#!/usr/bin/env python

import os
import sh
import random
import cv2
import pickle
import shutil
import glob


###############################################################################
#                                Helper Functions                             #
###############################################################################

def shuffle(df, n=1, axis=0):
    random.shuffle(df)
    return df


def check_frames_are_present(imgs, temp_dir=None):
    if len(imgs) == 0:
        if temp_dir:
            shutil.rmtree(temp_dir)
        return False
    return True


class FFMPEGNotFound(Exception):
    pass


def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


def burst_frames_to_shm(vid_path, shm_dir_path):
    """
    - To burst frames in a temporary directory in shared memory.
    - Directory name is chosen as random 128 bits so as to avoid clash
      during parallelization
    - Returns path to directory containing frames for the specific video
    """
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    target_mask = os.path.join(temp_dir, '%04d.jpg')
    if not check_ffmpeg_exists():
        raise FFMPEGNotFound()
    try:
        sh.ffmpeg('-i', vid_path,
                  '-q:v', str(1),
                  '-r', 8,
                  '-f', 'image2', target_mask)
    except Exception as e:
        print(repr(e))
    return temp_dir


def burst_video_into_frames(vid_path, shm_dir_path):
    temp_dir = burst_frames_to_shm(vid_path, shm_dir_path)
    imgs = find_images_in_folder(temp_dir, formats=['jpg'])
    assert check_frames_are_present(imgs, temp_dir), \
        "not frames bursted in {}...".format(vid_path)
    return temp_dir, imgs


def resize_images(imgs, img_size=-1):
    for img in imgs:
        img = cv2.imread(img)
        if img_size > 0:
            img = resize_by_short_edge(img, img_size)
        yield img


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


###############################################################################
#                                 File management                             #
###############################################################################

def ensure_output_dir_exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def clear_temp_dir(temp_dir):
    shutil.rmtree(temp_dir)


def dump_in_pickel(data, output_folder, filename):
    pickle.dump(data, open(output_folder + '/' + filename + '.pkl', 'wb'))


###############################################################################
#                       Helper Functions for input iterator                   #
###############################################################################

def find_images_in_folder(folder, formats=['jpg', 'png']):
    images = []
    for format_ in formats:
        files = glob.glob('{}/*.{}'.format(folder, format_))
        images.extend(files)
    return sorted(images)


def get_single_video_path(folder_name):
    video_filenames = glob.glob("{}/*.mp4".format(folder_name))
    assert len(video_filenames) == 1
    return video_filenames[0]
