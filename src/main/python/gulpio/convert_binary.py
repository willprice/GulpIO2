import os
import shutil
import glob
import cv2
import pickle

from gulpio.utils import resize_by_short_edge, shuffle, burst_frames_to_shm
from gulpio.parse_input import Input_from_csv, Input_from_json
from gulpio.gulpio import GulpVideoIO


def initialize_filenames(output_folder, chunk_no):
    bin_file_path = os.path.join(output_folder, 'data{}.bin'.format(chunk_no))
    meta_file_path = os.path.join(output_folder, 'meta{}.bin'.format(chunk_no))
    return bin_file_path, meta_file_path


def get_video_as_label_and_frames(entry, video_path, labels2idx):
    video_id = entry['id_'] #TODO
    label = entry['template'] #TODO
    folder = create_folder_name(video_id,
                                video_path=video_path,
                                label=label,
                                start_t=entry['start_time'],
                                end_t=entry['end_time'])
    imgs = find_images_in_folder(folder)
    if not check_frames_are_present(imgs):
        if check_one_mp4_file_present(get_video_path(folder)):
            imgs = burst_video_into_frames(get_video_path(folder)[0],
                                           shm_dir_path)
        else:
            print("neither video nor frames are present in {}".format(folder))
    print(labels2idx)
    if not labels2idx == None:
        label_idx = labels2idx[label]
    else:
        label_idx = -1
    return video_id, imgs, label_idx, label

def create_folder_name(video_id, video_path=None, label=None, start_t=None,
                       end_t=None):
    if video_path:
        print(video_path, video_id)
        return os.path.join(video_path, video_id)
    return os.path.join(args.frames_path,
                        label,
                        video_id) + "_{:06d}_{:06d}".format(start_t, end_t)



def find_images_in_folder(folder, formats=['jpg', 'png']):
    images = []
    for format_ in formats:
        files = glob.glob('{}/*.{}'.format(folder, format_))
        images.extend(files)
    return sorted(images)


def check_frames_are_present(imgs, temp_dir=None):
    if len(imgs) == 0:
        print("No frames present...")
        if temp_dir:
            shutil.rmtree(temp_dir)
        return False
    return True


def get_video_path(folder_name):
    return glob.glob(folder_name + "*.mp4")


def check_one_mp4_file_present(vid_path):
    if len(vid_path) > 1:
        print("more than one video file in {}".format(vid_path))
        return False
    if len(vid_path) == 0 or not os.path.isfile(vid_path[0]):
        print("no video file {}".format(vid_path))
        return False
    return True


def get_resized_image(imgs, img_size):
    for img in imgs:
        img = cv2.imread(img)
        img = resize_by_short_edge(img, img_size)
        yield img


def burst_video_into_frames(vid_path, shm_dir_path):
    temp_dir = burst_frames_to_shm(vid_path, shm_dir_path)
    imgs = sorted(glob.glob(temp_dir + '/*.jpg'))
    check_frames_are_present(imgs, temp_dir)
    clear_temp_dir(temp_dir)
    return imgs

def clear_temp_dir(temp_dir):
    shutil.rmtree(temp_dir)

def create_chunk(inputs, path, labels2idx):
    df, output_folder, chunk_no, img_size = inputs
    bin_file_path, meta_file_path = initialize_filenames(output_folder,
                                                         chunk_no)
    gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
    gulp_file.open()
    for idx, row in enumerate(df):
        video_id, imgs, label_idx, label = get_video_as_label_and_frames(row,
                                                                         path,
                                                                         labels2idx)
        #ensure_frames_are_present(imgs)
        [gulp_file.write(label_idx, video_id, img)
            for img in get_resized_image(imgs, img_size)]
    gulp_file.close()
    return True


def dump_labels2idx_in_pickel(labels2idx, output_folder):
    pickle.dump(labels2idx, open(output_folder + '/label2idx.pkl', 'wb'))

def get_shuffled_data(input_csv, input_json, labels_available,
                      output_folder=None):
    # read data
    if input_csv:
        data_object = Input_from_csv(input_csv)
    elif input_json:
        data_object = Input_from_json(input_json)
    # create label to idx map
    print(labels_available)
    labels2idx = None
    if labels_available:
        labels2idx = data_object.label2idx
        print(" > Creating label dictionary")
        dump_labels2idx_in_pickel(labels2idx, output_folder)

    data = data_object.get_data()
    # shuffle df and write binary file
    print(" > Shuffling data list")
    return shuffle(data), labels2idx

def compute_number_of_chunks(data, videos_per_chunk):
    return len(data) // videos_per_chunk + 1


def distribute_data_in_chunks(data, videos_per_chunk, output_folder, img_size):
    num_chunks = compute_number_of_chunks(data, videos_per_chunk)
    # set input array
    print(" > Setting up data chunks")
    inputs = []

    for chunk_id in range(num_chunks):
        if chunk_id == num_chunks - 1:
            df_sub = data[chunk_id * videos_per_chunk:]
        else:
            df_sub = data[chunk_id * videos_per_chunk:
                          (chunk_id + 1) * videos_per_chunk]
        input_data = [df_sub, output_folder, chunk_id, img_size]
        inputs.append(input_data)

    return inputs

def get_chunked_input(input_csv, input_json, videos_per_chunk, output_folder,
                      img_size, dump_labels2idx):
    data, labels2idx = get_shuffled_data(input_csv, input_json, dump_labels2idx,
                             output_folder)
    return distribute_data_in_chunks(data, videos_per_chunk, output_folder,
                                     img_size), labels2idx
