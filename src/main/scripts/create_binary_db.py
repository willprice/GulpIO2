#!/usr/bin/env python

"""Create a binary file including all video frames with RecordIO convention.
Usage:
    create_binary from_video [<videos_path>] [<csv_file>|<input_json>] [<output_folder>] [<video_per_chunk>] [<image_size>] [<temp_dir>] [<labels>]
    create_binary from_frames [<frames_path>] [<csv_file>|<input_json>] [<output_folder>] [<video_per_chunk>] [<image_size>] [<labels>]
    create_binary (-h | --help)
    create_binary --version
Options:
    -h --help     Show this screen.
    --version     Show version.
    json_file     paths to json files
"""

import os
import pickle
import glob
import cv2
import pandas as pd
import numpy as np
from docopt import docopt

from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

from gulpio.gulpio import GulpVideoIO
from gulpio.convert_binary import get_chunked_input
from gulpio.utils import resize_by_short_edge, shuffle, burst_frames_to_shm
from gulpio.parse_input import Input_from_csv, Input_from_json

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
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
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
    arguments = docopt(__doc__)
    if arguments['from_video']:
        videos_path = arguments['<video_path>'] # help=('Path to videos'))
        shm_dir_path = arguments['<temp_dir>']
    if arguments['from_frames']:
        frames_path = arguments['<frames_path>']
    input_csv = arguments['<input_csv>']
    input_json = arguments['<input_json>'] #  help=('path to the json file to convert the videos for (train/validation/test)'))
    output_folder = arguments['<output_folder>'] #  help='Output folder')
    vid_per_chunk = arguments['<videos_per_chunk>'] # help='number of videos in a chunk')
    num_workers = arguments['<num_workers>'] # help='number of workers.')
    img_size = arguments['image_size'] # help='shortest img size to resize all input images.')
    dump_label2idx = arguments['<label>']

    # create output folder if not there
    ensure_output_dir_exists(output_folder)

    inputs = get_chunked_input(input_csv, input_json, vid_per_chunk,
                               output_folder, img_size)

    #parallel_process(inputs, create_chunk, n_jobs=args.num_workers)
    results = Parallel(n_jobs=num_workers)(delayed(create_chunk)(i)
                                           for i in tqdm(inputs))
