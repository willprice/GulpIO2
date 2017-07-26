#!/usr/bin/env python

"""Create a binary file including all video frames with RecordIO convention.
Usage:
    create_binary [--videos_path <videos_path>]
                  [--input_csv <input_csv>|--input_json <input_json>]
                  [--output_folder <output_folder>]
                  [--videos_per_chunk <videos_per_chunk>]
                  [--num_workers <num_workers>]
                  [--image_size <image_size>]
    create_binary (-h | --help)
    create_binary --version

Options:
    -h --help                               Show this screen.
    --version                               Show version.
    --videos_path=<videos_path>             Path to video files
    --frames_path=<frames_path>             Path to frames files
    --input_csv=<input_csv>                 csv file with information about videos
    --input_json=<input_json>               json file with information about videos
    --output_folder=<output_folder>         Output folder for binary files
    --videos_per_chunk=<videos_per_chunk>   Number of videos in one chunk
    --num_workers=<num_workers>             Number of parallel processes
    --image_size=<image_size>               Size of smaller edge of resized frames
"""

import os
import sys
import pickle
from docopt import docopt
from tqdm import tqdm
from joblib import Parallel, delayed

from gulpio.parse_input import (Input_from_csv,
                                Input_from_json,
                               )
from gulpio.convert_binary import (Chunking,
                                   WriteChunks,
                                  )
from gulpio.utils import (ensure_output_dir_exists,
                          dump_in_pickel,
                          clear_temp_dir,
                         )


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)

    videos_path = arguments['--videos_path']
    output_folder = arguments['--output_folder']

    vid_per_chunk = int(arguments['--videos_per_chunk'])
    num_workers = int(arguments['--num_workers'])
    img_size = int(arguments['--image_size'])

    input_csv = arguments['--input_csv']
    input_json = arguments['--input_json']

    shm_dir_path = "temp_bursted_frames"

    if input_csv:
        data_object = Input_from_csv(input_csv)
    elif input_json:
        data_object = Input_from_json(input_json, videos_path)
    # data, labels2idx = data_object.get_data()

    iter_data = data_object.iter_data()

    chunks = Chunking(iter_data, vid_per_chunk)

    # create output folder if not there
    ensure_output_dir_exists(output_folder)

    # save label to index dictionary if it exists
    #if not labels2idx == {}:
    #    dump_in_pickel(labels2idx, output_folder, 'label2idx')
    chunk_writer = WriteChunks({}, img_size, output_folder, shm_dir_path)

    #
    for chunk in chunks:
        chunk_writer.write_chunk(chunk)
    sys.exit()
    results = Parallel(n_jobs=num_workers)(delayed(chunk_writer.write_chunk)(i,
                                                                {},
                                                                img_size,
                                                                output_folder,
                                                                shm_dir_path
                                                                )
                                            for i in chunks)
    clear_temp_dir(shm_dir_path)
