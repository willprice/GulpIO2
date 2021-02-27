#!/usr/bin/env python

import os
from io import BytesIO
import PIL.Image

import numpy as np
import sh
import random
import shutil
import glob
from contextlib import contextmanager

###############################################################################
#                                Helper Functions                             #
###############################################################################
from typing import Iterable, Iterator, Union

import simplejpeg


class FFMPEGNotFound(Exception):
    pass


# If you update this, then the system_tests.py expected sizes will likely change.
_JPEG_WRITE_QUALITY = 90

def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


@contextmanager
def temp_dir_for_bursting(shm_dir_path='/dev/shm'):
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    yield temp_dir
    shutil.rmtree(temp_dir)


def img_to_jpeg_bytes(img: np.ndarray) -> bytes:
    if img.ndim == 2:
        colorspace = "Gray"
        img = img[..., None]
    elif img.ndim == 3:
        colorspace = "RGB"
    else:
        raise ValueError("Unsupported img shape: {}".format(img.shape))
    return simplejpeg.encode_jpeg(img, quality=_JPEG_WRITE_QUALITY, colorspace=colorspace)


def jpeg_bytes_to_img(jpeg_bytes: bytes) -> np.ndarray:
    colorspace = simplejpeg.decode_jpeg_header(jpeg_bytes)[2]
    # 3-channel jpegs are internally encoded as YCbCr and we have to impose our desired
    # colorspace conversion which we assume is RGB.
    if colorspace == "YCbCr":
        colorspace = "RGB"
    img = simplejpeg.decode_jpeg(
            jpeg_bytes, fastdct=True, fastupsample=True, colorspace=colorspace
    )
    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    return img


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


class ImageNotFound(Exception):
    pass


class DuplicateIdException(Exception):
    pass


def resize_images(imgs: Iterable[str], img_size=-1) -> Iterator[np.ndarray]:
    for img in imgs:
        img_path = img
        img = PIL.Image.open(img_path)
        if img is None:
            raise ImageNotFound("Image is  None from path:{}".format(img_path))
        if img_size > 0:
            img = resize_by_short_edge(img, img_size)
        else:
            img = np.asarray(img)
        yield img


def resize_by_short_edge(
    img: Union[str, PIL.Image.Image, np.ndarray],
    size: int
) -> np.ndarray:
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)
    elif isinstance(img, str):
        img_path = img
        img = PIL.Image.open(img_path)
        if img is None:
            raise ImageNotFound("Image read None from path ", img_path)
    if size < 1:
        return np.asarray(img)
    w, h = img.width, img.height
    if h < w:
        scale = w / float(h)
        new_width = int(size * scale)
        img = img.resize((new_width, size), PIL.Image.BILINEAR)
    else:
        scale = h / float(w)
        new_height = int(size * scale)
        img = img.resize((size, new_height), PIL.Image.BILINEAR)
    return np.asarray(img)


def remove_entries_with_duplicate_ids(output_directory, meta_dict):
    meta_dict = _remove_duplicates_in_metadict(meta_dict)
    from gulpio2.fileio import GulpDirectory
    gulp_directory = GulpDirectory(output_directory)
    existing_ids = list(gulp_directory.merged_meta_dict.keys())
    # this assumes no duplicates in existing_ids
    new_meta = []
    for meta_info in meta_dict:
        if str(meta_info['id']) in existing_ids:
            print('Id {} already in GulpDirectory, I skip it!'
                  .format(meta_info['id']))
        else:
            new_meta.append(meta_info)
    if len(new_meta) == 0:
        print("no items to add... Abort")
        raise DuplicateIdException
    return new_meta


def _remove_duplicates_in_metadict(meta_dict):
    ids = list(enumerate(map(lambda d: d['id'], meta_dict)))
    if len(set(map(lambda d: d[1], ids))) == len(ids):
        return meta_dict
    else:
        new_meta = []
        seen_id = []
        for index, id_ in ids:
            if id_ not in seen_id:
                new_meta.append(meta_dict[index])
                seen_id.append(id_)
            else:
                print('Id {} more than once in json file, I skip it!'
                      .format(id_))
        return new_meta


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
