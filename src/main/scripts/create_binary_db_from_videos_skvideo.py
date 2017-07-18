import os
import argparse
import pickle
import cv2
import pandas as pd
import numpy as np
import skvideo.io

from tqdm import tqdm
from joblib import Parallel, delayed

from gulpio import GulpVideoIO
from gulpio import resize_by_short_edge, shuffle


def create_chunk(inputs):
    df = inputs[0]
    output_folder = inputs[1]
    chunk_no = inputs[2]
    img_size = inputs[3]
    bin_file_path = os.path.join(output_folder, 'data{}.bin'.format(chunk_no))
    meta_file_path = os.path.join(output_folder, 'meta{}.bin'.format(chunk_no))
    gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
    gulp_file.open()
    for idx, row in df.iterrows():
        video_id = row.youtube_id
        label = row.label
        start_t = row.time_start
        end_t = row.time_end
        folder_name = os.path.join(
            args.videos_path, label, video_id) + "_{:06d}_{:06d}".format(start_t, end_t)
        vid_path = folder_name + ".mp4"

        outputparameters = {"-r": "%d" % 8,
                            "-q:v": "%d" % 1}
        try:
            if not os.path.isfile(vid_path):
                print("Path doesn't exists ...")
                continue
            vid_stats = skvideo.io.ffprobe(vid_path)
            if not vid_stats or 'video' not in vid_stats:
                print("No video metadata ...")
                continue
            if float(vid_stats['video']['@duration']) < 0.5:
                print("Duration less than 0.5 secs ...")
                continue
            reader = skvideo.io.FFmpegReader(vid_path,
                                             outputdict=outputparameters)
            for img in reader.nextFrame():
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = resize_by_short_edge(img, img_size)
                label_idx = labels2idx[label]
                gulp_file.write(label_idx, video_id, img)
        except Exception as e:
            error_out = repr(e)
            if not error_out == "RuntimeError('',)":
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
    args = p.parse_args()

    # read data csv list
    print(" > Reading data list (csv)")
    df = pd.read_csv(args.input_csv)

    # create output folder if not there
    os.makedirs(args.output_folder, exist_ok=True)

    # create label to idx map
    print(" > Creating label dictionary")
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

    results = Parallel(n_jobs=args.num_workers)(delayed(create_chunk)(i) for i in tqdm(inputs))
