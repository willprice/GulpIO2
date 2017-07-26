import os
import shutil

from gulpio.utils import (get_resized_image,
                          burst_video_into_frames,
                          )
from gulpio.gulpio import GulpVideoIO


def check_frames_are_present(imgs, temp_dir=None):
    if len(imgs) == 0:
        if temp_dir:
            shutil.rmtree(temp_dir)
        return False
    return True


class WriteChunks(object):

    def __init__(self, labels2idx, img_size, output_folder, shm_dir_path):
        self.labels2idx = labels2idx
        self.img_size = img_size
        self.output_folder = output_folder
        self.shm_dir_path = shm_dir_path
        self.count_chunks = 0

    def initialize_filenames(self, output_folder, chunk_no):
        bin_file_path = os.path.join(output_folder,
                                     'data{}.bin'.format(chunk_no))
        meta_file_path = os.path.join(output_folder,
                                      'meta{}.bin'.format(chunk_no))
        return bin_file_path, meta_file_path

    def detect_input(self, input_files):
        if len(input_files) == 0:
            return 'missing'  # TODO
        if ('mp4' in input_files[0] and len(input_files) == 1):
            return 'video'
        elif all(('jpg' in i or 'png' in i) for i in input_files):
            return 'frames'

    def write_chunk(self, input_chunk):
        bin_file_path, meta_file_path = self.initialize_filenames(
            self.output_folder,
            self.count_chunks)
        gulp_file = GulpVideoIO(bin_file_path, 'wb', meta_file_path)
        gulp_file.open()
        for row in input_chunk:
            files = row['files']
            input_type = self.detect_input(files)
            if input_type == 'video':
                files = burst_video_into_frames(files[0], self.shm_dir_path)
            imgs = get_resized_image(files, self.img_size)

            # meta_information = row['meta']  # TODO include in Gulpio
            id_ = row['id']
            [gulp_file.write(self.count_chunks, id_, img) for img in imgs]

            self.count_chunks += 1
        gulp_file.close()
        return True


class Chunking:

    def __init__(self, iter_data, videos_per_chunk):
        self.iter_data = iter_data
        self.videos_per_chunk = videos_per_chunk
        self.iter_data_element = next(self.iter_data)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        if not self.iter_data_element:
            raise StopIteration()
        count = 0
        while (self.iter_data_element and count < self.videos_per_chunk):
            chunk.append(self.iter_data_element)
            self.iter_data_element = next(self.iter_data)
            count += 1
        return chunk
