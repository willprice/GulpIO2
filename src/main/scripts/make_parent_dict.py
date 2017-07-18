import os
import glob
import argparse
import pickle

from collections import namedtuple

ItemInfo = namedtuple('ItemInfo', ['chunk_id', 'youtube_id'])


def make_dict(data_path):
    meta_files_list = sorted(glob.glob(os.path.join(data_path, "meta*")))
    target_meta_file_path = os.path.join(data_path, "parent_dict.pkl")
    target_dict = {}
    counter = 0
    for i, meta_file in enumerate(meta_files_list):
        meta_dict = pickle.load(open(meta_file, 'rb'))
        for k in meta_dict.keys():
            info = ItemInfo(i, k)
            target_dict[counter] = info
            counter += 1
    print("Total number of elements in the dictionary = {}".format(len(target_dict)))
    pickle.dump(target_dict, open(target_meta_file_path, "wb"))


if __name__ == '__main__':
    description = 'Create a dictionary to map indices to chunks.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('binary_data_path', type=str,
                   help=('Path to binarized data'))
    args = p.parse_args()
    make_dict(args.binary_data_path)
