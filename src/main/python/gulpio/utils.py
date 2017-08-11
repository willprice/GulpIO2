#!/usr/bin/env python

import os
import sh
import random
import cv2
import shutil
import glob
from contextlib import contextmanager


###############################################################################
#                                Helper Functions                             #
###############################################################################


class FFMPEGNotFound(Exception):
    pass


def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


@contextmanager
def temp_dir_for_bursting(shm_dir_path='/dev/shm'):
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    yield temp_dir
    shutil.rmtree(temp_dir)


def burst_frames_to_shm(vid_path, temp_burst_dir, frame_rate=None):
    """
    - To burst frames in a temporary directory in shared memory.
    - Directory name is chosen as random 128 bits so as to avoid clash
      during parallelization
    - Returns path to directory containing frames for the specific video
    """
    target_mask = os.path.join(temp_burst_dir, '%04d.jpg')
    if not check_ffmpeg_exists():
        raise FFMPEGNotFound()
    try:
        ffmpeg_args = [
            '-i', vid_path,
            '-q:v', str(1),
            '-f', 'image2',
            target_mask,
        ]
        if frame_rate:
            ffmpeg_args.insert(2, '-r')
            ffmpeg_args.insert(3, frame_rate)
        sh.ffmpeg(*ffmpeg_args)
    except Exception as e:
        print(repr(e))


def burst_video_into_frames(vid_path, temp_burst_dir, frame_rate=None):
    burst_frames_to_shm(vid_path, temp_burst_dir, frame_rate=frame_rate)
    return find_images_in_folder(temp_burst_dir, formats=['jpg'])


def resize_images(imgs, img_size=-1):
    for img in imgs:
        img = cv2.imread(img)
        if img_size > 0:
            img = resize_by_short_edge(img, img_size)
        yield img


def resize_by_short_edge(img, size):
    if isinstance(img, str) or isinstance(img, unicode):
        img = cv2.imread(img)
    if size < 1:
        return img
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


###############################################################################
#                       Helper Functions for input iterator                   #
###############################################################################

def find_images_in_folder(folder, formats=['jpg']):
    images = []
    for format_ in formats:
        files = glob.glob('{}/*.{}'.format(folder, format_))
        images.extend(files)
    return sorted(images)


def get_single_video_path(folder_name, format_='mp4'):
    video_filenames = glob.glob("{}/*.{}".format(folder_name, format_))
    assert len(video_filenames) == 1
    return video_filenames[0]
