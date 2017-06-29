import os
import sys
import argparse
import pickle
import glob
import sh
import shutil
import random
import cv2
import pandas as pd
import numpy as np

sys.path.insert(0, "../")

from tqdm import tqdm
from pprint import pprint
from gulpio import GulpVideoIO
from joblib import Parallel, delayed


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
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
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
          '-threads', 1,
                  '-r', 20,
                  '-f', 'image2', target_mask)
    except Exception as e:
        print(repr(e))
    return temp_dir


def optical_flow(imgs):

    num_avg_frames = np.ceil(20 / 8)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    # print("Avg number of frames : {0}".format(num_avg_frames))

    # read an init frame
    frame1 = imgs[0]
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # set a matrix to keep averagte optical flow for last k frames
    avg_of_frame = np.zeros(
        [frame1.shape[0], frame1.shape[1], 2], dtype=np.float32)
    avg_count = 1
    c = 0
    save_count = 1

    out_imgs = []
    for img in imgs[1:]:
        c += 1
        frame2 = img
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next, None, 0.5, 3, 12, 3, 5, 1.2, 0)
        flow[..., 0] = flow[..., 0] - np.mean(flow[..., 0])
        flow[..., 1] = flow[..., 1] - np.mean(flow[..., 1])
        avg_of_frame += flow
        avg_count += 1

        if avg_count == num_avg_frames:
            avg_count = 1
            flow = avg_of_frame / num_avg_frames
            avg_of_frame = np.zeros(
                [frame1.shape[0], frame1.shape[1], 2], dtype=np.float32)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            out_imgs.append(rgb)
            save_count += 1
        prvs = next
    return out_imgs


def create_chunk(inputs, shm_dir_path, is_test):
    df = inputs[0]
    output_folder = inputs[1]
    chunk_no = inputs[2]
    img_size = inputs[3]
    bin_file_path = os.path.join(output_folder, 'data%03d.bin' % chunk_no)
    meta_file_path = os.path.join(output_folder, 'meta%03d.bin' % chunk_no)
    gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
    gulp_file.open()
    for idx, row in df.iterrows():
        video_id = row.youtube_id
        if not is_test:
            label = row.label
        start_t = row.time_start
        end_t = row.time_end
        if not is_test:
            folder_name = os.path.join(
                args.videos_path, label, video_id) + "_{:06d}_{:06d}".format(start_t, end_t)
            label_idx = labels2idx[label]
        else:
            folder_name = os.path.join(
                args.videos_path, video_id) + "_{:06d}_{:06d}".format(start_t, end_t)
            label_idx = None
        vid_path = folder_name + ".mp4"

        if not os.path.isfile(vid_path):
            print("Path doesn't exists for {}".format(vid_path))
            continue
        temp_dir = burst_frames_to_shm(vid_path, shm_dir_path)
        imgs = sorted(glob.glob(temp_dir + '/*.jpg'))
        imgs_array = []
        if len(imgs) == 0:
            print("No images bursted ...")
            shutil.rmtree(temp_dir)
            continue
        try:
            for img in imgs:
                img = cv2.imread(img)
                img = resize_by_short_edge(img, img_size)
                imgs_array.append(img)
        except Exception as e:
            print(repr(e))
        shutil.rmtree(temp_dir)
        flow_array = optical_flow(imgs_array)
        # sample_img_path = "sample_imgs"
        try:
            for i, img in enumerate(flow_array):
                # cv2.imwrite(os.path.join(sample_img_path, "frame%03d.jpg" % i), img)
                gulp_file.write(label_idx, video_id, img)
        except Exception as e:
            print(repr(e))

    gulp_file.close()


if __name__ == '__main__':
    description = 'Create a binary file including all video frames with RecordIO convention.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('videos_path', type=str,
                   help=('Path to videos'))
    p.add_argument('input_csv', type=str,
                   help=('Kinetics CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
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
    p.add_argument('--is_test', type=int, default=0,
           help='is given data test set')
    args = p.parse_args()

    # read data csv list
    print(" > Reading data list (csv)")
    df = pd.read_csv(args.input_csv)

    # create output folder if not there
    os.makedirs(args.output_folder, exist_ok=True)

    # create label to idx map
    print(" > Creating label dictionary")
    if not args.is_test:
        labels = sorted(pd.unique(df['label']))
        assert len(labels) == 400
        labels2idx = {}
        label_counter = 0
        for label in labels:
            labels2idx[label] = label_counter
            label_counter += 1
        pickle.dump(labels2idx, open(args.output_folder + '/label2idx.pkl', 'wb'))

    # shuffle df and write binary file
    print(" > Shuffling data list")
    df = shuffle(df)

    # set input array
    inputs = []
    for idx, df_sub in df.groupby(np.arange(len(df)) // args.vid_per_chunk):
        input_data = [df_sub, args.output_folder, idx, args.img_size]
        inputs.append(input_data)

    results = Parallel(n_jobs=args.num_workers)(delayed(create_chunk)(i, args.shm_dir_path, args.is_test) for i in tqdm(inputs))
