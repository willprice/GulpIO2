import os
import argparse
import pickle
import glob
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from gulpio import GulpVideoIO
from concurrent.futures import ProcessPoolExecutor, as_completed


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
            args.frames_path, label, video_id) + "_{:06d}_{:06d}".format(start_t, end_t)
        imgs = glob.glob(folder_name + '/*.jpg')
        for img in imgs:
            img = cv2.imread(img)
            img = resize_by_short_edge(img, img_size)
            label_idx = labels2idx[label]
            gulp_file.write(label_idx, video_id, img)
    gulp_file.close()


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=0):
    """
        A parallel version of the map function with a progress bar. 
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a)
                 for a in array[:front_num]]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for
    # benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


if __name__ == '__main__':
    description = 'Create a binary file including all video frames with RecordIO convention.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('frames_path', type=str,
                   help=('Path to bursted frames'))
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
    parallel_process(inputs, create_chunk, n_jobs=args.num_workers)
