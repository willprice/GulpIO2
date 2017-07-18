import os

import argparse
import pickle
import glob
import sh
import shutil
import random
import cv2
import numpy as np

from tbntools import data
from tqdm import tqdm
from pprint import pprint
from gulpio import GulpVideoIO
from joblib import Parallel, delayed


# def resize_by_short_edge(img, size):
#     h, w = img.shape[0], img.shape[1]
#     if h < w:
#         scale = w / float(h)
#         new_width = int(size * scale)
#         img = cv2.resize(img, (new_width, size))
#     else:
#         scale = h / float(w)
#         new_height = int(size * scale)
#         img = cv2.resize(img, (size, new_height))
#     return img
# 
# 
# def shuffle(df, n=1, axis=0):
#     df = df.copy()
#     for _ in range(n):
#         df.apply(np.random.shuffle, axis=axis)
#     return df
# 
# 
# def burst_frames_to_shm(vid_path, shm_dir_path):
#     """
#     - To burst frames in a temporary directory in shared memory.
#     - Directory name is chosen as random 128 bits so as to avoid clash during
#       parallelization
#     - Returns path to directory containing frames for the specific video
#     """
#     hash_str = str(random.getrandbits(128))
#     temp_dir = os.path.join(shm_dir_path, hash_str)
#     os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
#     target_mask = os.path.join(temp_dir, '%04d.jpg')
#     try:
#         sh.ffmpeg('-i', vid_path,
#                   '-q:v', str(1),
#                   '-r', 8,
#                   '-threads', 1,
#                   '-f', 'image2', target_mask)
#     except Exception as e:
#         print(repr(e))
#     return temp_dir


def create_chunk(inputs, shm_dir_path):
    df = inputs[0]
    output_folder = inputs[1]
    chunk_no = inputs[2]
    img_size = inputs[3]
    bin_file_path = os.path.join(output_folder, 'data%03d.bin' % chunk_no)
    meta_file_path = os.path.join(output_folder, 'meta%03d.bin' % chunk_no)
    gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
    gulp_file.open()
    for idx, row in enumerate(df):
        video_id = str(row['id'])
        label = row['template']

        folder_name = os.path.join(args.videos_path, video_id)
        vid_path = glob.glob(os.path.join(folder_name, "*.mp4"))
        assert len(vid_path) == 1
        vid_path = vid_path[0]

        if not os.path.isfile(vid_path):
            print("Path doesn't exists for {}".format(vid_path))
            continue
        temp_dir = burst_frames_to_shm(vid_path, shm_dir_path)
        imgs = sorted(glob.glob(temp_dir + '/*.jpg'))
        if len(imgs) == 0:
            print("No images bursted ...")
            shutil.rmtree(temp_dir)
            continue
        try:
            for img in imgs:
                img = cv2.imread(img)
                img = resize_by_short_edge(img, img_size)
                label_idx = labels2idx[label]
                gulp_file.write(label_idx, video_id, img)
        except Exception as e:
            print(repr(e))
        shutil.rmtree(temp_dir)
    gulp_file.close()


if __name__ == '__main__':
    description = 'Create binaries for 20BN dataset with RecordIO convention.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('videos_path', type=str,
                   help=('Path to videos'))
    p.add_argument('input_json', type=str,
                   help=('path to the json file to convert the videos for '
                         '(train/validation/test)'))
    p.add_argument('output_folder', type=str,
                   help='Output folder')
    p.add_argument('vid_per_chunk', type=int,
                   help='number of videos in a chunk')
    p.add_argument('num_workers', type=int,
                   help='number of workers.')
    p.add_argument('img_size', type=int,
                   help='shortest img size to resize all input images.')
    p.add_argument('shm_dir_path', type=str,
                   help='path to the temp directory in shared memory.')
    args = p.parse_args()

    # read json using tbntools
    data_storage = data.RawDataset.load(args.input_json, label='template').storage

    # create output folder if not there
    os.makedirs(args.output_folder, exist_ok=True)

    # create label to idx map
    print(" > Creating label dictionary")
    labels = sorted(set([item['template'] for item in data_storage]))
    print("Found {} labels".format(len(labels)))
    labels2idx = {}
    label_counter = 0
    for label in labels:
        labels2idx[label] = label_counter
        label_counter += 1
    pickle.dump(labels2idx, open(args.output_folder + '/label2idx.pkl', 'wb'))

    # set input array
    inputs = []
    num_chunks = len(data_storage) // args.vid_per_chunk + 1
    for chunk_id in range(num_chunks):
        if chunk_id == num_chunks - 1:
            df_sub = data_storage[chunk_id * args.vid_per_chunk:]
        else:
            df_sub = data_storage[chunk_id * args.vid_per_chunk: (chunk_id + 1) * args.vid_per_chunk]
        input_data = [df_sub, args.output_folder, chunk_id, args.img_size]
        inputs.append(input_data)

    results = Parallel(n_jobs=args.num_workers)(delayed(create_chunk)(i, args.shm_dir_path) for i in tqdm(inputs))
