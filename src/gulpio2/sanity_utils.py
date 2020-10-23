import os
import collections


###############################################################################
#                           Checks                                            #
###############################################################################

def check_meta_file_size_larger_zero(gulp_directory):
    return [chunk.meta_file_path
            for chunk in gulp_directory.chunks()
            if os.stat(chunk.meta_file_path).st_size == 0]


def check_data_file_size_larger_zero(gulp_directory):
    return [chunk.data_file_path
            for chunk in gulp_directory.chunks()
            if os.stat(chunk.data_file_path).st_size == 0]


def check_data_file_size(gulp_directory):
    result = []
    for chunk in gulp_directory.chunks():
        last_frame_info = (chunk.meta_dict[next(reversed(chunk.meta_dict))]
                                          ['frame_info']
                                          [-1])
        data_file_size_from_meta = last_frame_info[0] + last_frame_info[2]
        data_file_size = os.stat(chunk.data_file_path).st_size
        if not data_file_size == data_file_size_from_meta:
            result.append(chunk.data_file_path)
    return result


def check_for_duplicate_ids(gulp_directory):
    return get_duplicate_entries(
        extract_all_ids(gulp_directory))


###############################################################################
#                           Helper Functions                                  #
###############################################################################

def extract_all_ids(gulp_directory):
    all_ids = []
    for chunk in gulp_directory.chunks():
        all_ids.extend(chunk.meta_dict.keys())
    return all_ids


def get_duplicate_entries(list_):
    c = collections.Counter(list_)
    return [i for i in c if c[i] > 1]


###############################################################################
#                           Print Results                                     #
###############################################################################

def check_for_failures(result):
    print("Sanity Check: {}".format(result["message"]))
    if len(result["failures"]):
        print("Test failed for: {}".format(result["failures"]))
    else:
        print("Test passed")
