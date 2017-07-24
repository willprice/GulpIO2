#!/usr/bin/env python

"""Create a binary file including all video frames with RecordIO convention.
Usage:
    create_binary [--videos_path <videos_path>]
                  [--input_csv <input_csv>|--input_json <input_json>]
                  [--output_folder <output_folder>]
                  [--videos_per_chunk <videos_per_chunk>]
                  [--num_workers <num_workers>]
                  [--image_size <image_size>]
                  [--temp_dir <temp_dir>]
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
    --temp_dir=<temp_dir>                   Name of temporary directory for bursted frames
"""

import os
import pickle
from docopt import docopt
from tqdm import tqdm
from joblib import Parallel, delayed

from gulpio.parse_input import (Input_from_csv,
                                Input_from_json,
                               )
from gulpio.convert_binary import (get_chunked_input,
                                   create_chunk,
                                  )
from gulpio.utils import (ensure_output_dir_exists,
                          dump_in_pickel,
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

    if input_csv:
        data_object = Input_from_csv(input_csv)
    elif input_json:
        data_object = Input_from_json(input_json)
    data, labels2idx = data_object.get_data()

    # create output folder if not there
    ensure_output_dir_exists(output_folder)

    # save label to index dictionary if it exists
    if not labels2idx == {}:
        dump_in_pickel(labels2idx, output_folder, 'label2idx')

    # transform input into chunks and resize frames
    inputs = get_chunked_input(data,
                               vid_per_chunk,
                               )

    #parallel_process(inputs, create_chunk, n_jobs=args.num_workers)
    results = Parallel(n_jobs=num_workers)(delayed(create_chunk)(i,
                                                                 videos_path,
                                                                 labels2idx,
                                                                 img_size,
                                                                 output_folder,
                                                                 )
                                           for i in tqdm(inputs))
