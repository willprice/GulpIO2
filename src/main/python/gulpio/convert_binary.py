import os
import cv2

from gulpio.utils import (burst_frames_to_shm,
                          find_images_in_folder,
                          check_frames_are_present,
                          resize_by_short_edge,
                          )
from gulpio.gulpio import GulpVideoIO


class WriteChunks(object):

    def __init__(self, img_size, output_folder, shm_dir_path):
        self.img_size = img_size
        self.output_folder = output_folder
        self.shm_dir_path = shm_dir_path
        self.count_videos = 0

    def initialize_filenames(self, output_folder, chunk_no):
        bin_file_path = os.path.join(output_folder,
                                     'data{}.bin'.format(chunk_no))
        meta_file_path = os.path.join(output_folder,
                                      'meta{}.bin'.format(chunk_no))
        img_info_path = os.path.join(output_folder,
                                     'img_info{}.bin'.format(chunk_no))
        return bin_file_path, img_info_path, meta_file_path

    def write_chunk(self, input_chunk):
        (bin_file_path,
         img_info_path,
         meta_file_path) = self.initialize_filenames(self.output_folder,
                                                     self.count_videos)
        gulp_file = GulpVideoIO(bin_file_path,
                                'wb',
                                meta_file_path,
                                img_info_path)
        gulp_file.open()
        for video_info in input_chunk:
            id_ = video_info['id']
            meta_information = video_info['meta']
            filenames = video_info['files']

            fg = FramesGenerator(filenames, id_)
            frames = fg.extract_frames(self.img_size, self.shm_dir_path)

            gulp_file.write_meta(self.count_videos, id_, meta_information)
            [gulp_file.write(self.count_videos, id_, img) for img in frames]

            self.count_videos += 1
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


class FramesGenerator(object):

    def __init__(self, filenames, id_):
        self.filenames = filenames
        self.id_ = id_
        self.input_type = self.detect_input(self.filenames)

    def extract_frames(self, img_size, shm_dir_path=''):
        if self.input_type == 'missing':
            print("no input files given for {}".format(self.id_))
        if self.input_type == 'video':
            self. filenames = self.burst_video_into_frames(self.filenames[0],
                                                           shm_dir_path)
        return self.get_resized_image(self.filenames, img_size)

    def detect_input(self, input_files):
        if len(input_files) == 0:
            return 'missing'
        if ('.mp4' in input_files[0] and len(input_files) == 1):
            # TODO improvable
            return 'video'
        elif all(('jpg' in i or 'png' in i) for i in input_files):
            # TODO improvable... Only jpg?
            return 'frames'

    def burst_video_into_frames(self, vid_path, shm_dir_path):
        temp_dir = burst_frames_to_shm(vid_path, shm_dir_path)
        imgs = find_images_in_folder(temp_dir, formats=['jpg'])
        if not (check_frames_are_present(imgs, temp_dir)):
            print("not frames bursted in {}...".format(vid_path))
        return imgs

    def get_resized_image(self, imgs, img_size=-1):
        for img in imgs:
            img = cv2.imread(img)
            if img_size > 0:
                img = resize_by_short_edge(img, img_size)
            yield img
