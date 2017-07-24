import os
import shutil
import glob
import cv2

from gulpio.utils import (resize_by_short_edge,
                          shuffle,
                          burst_frames_to_shm,
                          clear_temp_dir,
                         )
from gulpio.gulpio import GulpVideoIO


def initialize_filenames(output_folder, chunk_no):
    bin_file_path = os.path.join(output_folder, 'data{}.bin'.format(chunk_no))
    meta_file_path = os.path.join(output_folder, 'meta{}.bin'.format(chunk_no))
    return bin_file_path, meta_file_path


def extract_images_in_folder(folder, shm_dir_path):
    imgs = find_images_in_folder(folder)
    if not check_frames_are_present(imgs):
        if check_one_mp4_file_present(get_video_path(folder)):
            imgs = burst_video_into_frames(get_video_path(folder)[0],
                                           shm_dir_path)
        else:
            print("neither video nor frames are present in {}".format(folder))
    return imgs


def get_video_as_label_and_frames(entry, video_path, labels2idx, shm_dir_path):
    video_id = entry['id']
    label = entry['label']
    folder = create_folder_name(video_id,
                                video_path=video_path,
                                start_t=entry['start_time'],
                                end_t=entry['end_time'])
    imgs = extract_images_in_folder(folder, shm_dir_path)
    if not labels2idx == {}:
        label_idx = labels2idx[label]
    else:
        label_idx = -1
    return video_id, imgs, label_idx, label


def create_folder_name(video_id, video_path=None, start_t=None,
                       end_t=None):
    if not start_t and not end_t:
        return os.path.join(video_path, video_id)
    return os.path.join(video_path,
                        video_id) + "_{:06d}_{:06d}".format(start_t, end_t)



def find_images_in_folder(folder, formats=['jpg', 'png']):
    images = []
    for format_ in formats:
        files = glob.glob('{}/*.{}'.format(folder, format_))
        images.extend(files)
    return sorted(images)


def check_frames_are_present(imgs, temp_dir=None):
    if len(imgs) == 0:
        if temp_dir:
            shutil.rmtree(temp_dir)
        return False
    return True


def get_video_path(folder_name):
    return glob.glob("{}/*.mp4".format(folder_name))


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
    imgs = find_images_in_folder(temp_dir, formats=['jpg'])
    if not (check_frames_are_present(imgs, temp_dir)):
        print("not frames bursted in {}...".format(vid_path))
    # clear_temp_dir(temp_dir)
    return imgs

def write_chunk(input_chunk, path, labels2idx, img_size, output_folder,
                shm_dir_path):
    df, chunk_no = input_chunk
    bin_file_path, meta_file_path = initialize_filenames(output_folder,
                                                         chunk_no)
    gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
    gulp_file.open()
    for idx, row in enumerate(df):
        video_id, imgs, label_idx, label = get_video_as_label_and_frames(
            row,
            path,
            labels2idx,
            shm_dir_path
           )

        #ensure_frames_are_present(imgs)
        [gulp_file.write(label_idx, video_id, img)
            for img in get_resized_image(imgs, img_size)]
    gulp_file.close()
    return True


def compute_number_of_chunks(data, videos_per_chunk):
    return len(data) // videos_per_chunk + 1


def distribute_data_in_chunks(data, videos_per_chunk):
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
        inputs.append([df_sub, chunk_id])

    return inputs

def get_chunked_input(data, videos_per_chunk):
    data = shuffle(data)
    return distribute_data_in_chunks(data,
                                     videos_per_chunk,
                                     )
