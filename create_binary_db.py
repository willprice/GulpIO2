import os
import sys
import argparse
import glob
import pickle
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from gulp_io import GulpVideoIO

SMALLER_SIDE_SIZE = 320


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


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


description = 'Create a binary file including all video frames with RecordIO convention.'
p = argparse.ArgumentParser(description=description)
p.add_argument('frames_path', type=str,
               help=('Path to bursted frames'))
p.add_argument('input_csv', type=str,
               help=('Kinetics CSV file containing the following format: '
                     'YouTube Identifier,Start time,End time,Class label'))
p.add_argument('output_folder', type=str,
               help='Output folder')
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

# shiffle df and write binary file
print(" > Shuffling data list")
# df = shuffle(df)

gulp_file = GulpVideoIO(args.output_folder + '/train_data.bin',
                        'wb', args.output_folder + '/train_meta.pkl')
gulp_file.open()
label_count = 0
counter = 0
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    video_id = row.youtube_id
    label = row.label
    start_t = row.time_start
    end_t = row.time_end
    folder_name = os.path.join(
        args.frames_path, label, video_id) + "_{:06d}_{:06d}".format(start_t, end_t)
    imgs = glob.glob(folder_name + '/*.jpg')
    # imgs = range(82)
    for img in imgs:
        img = cv2.imread(img)
        img = resize_by_short_edge(img, SMALLER_SIDE_SIZE)
        # img = np.random.randint(0, 255, size=[320,320,3])
        label_idx = labels2idx[label]
        gulp_file.write(label_idx, video_id, img)

    counter += 1
    if counter == 5:
        break
gulp_file.close()
