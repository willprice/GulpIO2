#!/usr/bin/env python

import os
import cv2
import pickle
import numpy as np
from abc import ABC, abstractmethod
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image
from collections import namedtuple, defaultdict
from tqdm import tqdm

from .utils import ensure_output_dir_exists


ImgInfo = namedtuple('ImgInfo', ['loc',
                                 'pad',
                                 'length'])
MetaInfo = namedtuple('MetaInfo', ['id_',
                                   'meta_data'])


class AbstractSerializer(ABC):

    @abstractmethod
    def load(self, file_pointer):
        pass

    @abstractmethod
    def dump(self, thing, file_pointer):
        pass


class PickleSerializer(AbstractSerializer):

    def load(self, file_pointer):
        return pickle.load(file_pointer)

    def dump(self, thing, file_pointer):
        pickle.dump(thing, file_pointer)


class JSONSerializer(AbstractSerializer):

    def load(self, file_pointer):
        return json.load(file_pointer)

    def dump(self, thing, file_pointer):
        json.dump(thing, file_pointer)


pickle_serializer = PickleSerializer()
json_serializer = JSONSerializer()


class GulpVideoIO(object):

    def __init__(self, path, flag, meta_path, img_info_path,
                 serializer=json_serializer):
        self.path = path
        self.flag = flag
        self.meta_path = meta_path
        self.img_info_path = img_info_path
        self.serializer = serializer

        self.is_open = False
        self.is_writable = False
        self.f = None
        self.img_dict = None
        self.meta_dict = None

    def get_or_create_dict(self, path):
        if os.path.exists(path):
            return self.serializer.load(open(path, 'r'))
        return defaultdict(list)

    def open(self):
        self.meta_dict = self.get_or_create_dict(self.meta_path)
        self.img_dict = self.get_or_create_dict(self.img_info_path)

        if self.flag == 'wb':
            self.f = open(self.path, self.flag)
            self.is_writable = True
        elif self.flag == 'rb':
            self.f = open(self.path, self.flag)
            self.is_writable = False
        self.is_open = True

    def close(self):
        if self.is_open:
            self.serializer.dump(self.meta_dict, open(self.meta_path, 'w'))
            self.serializer.dump(self.img_dict, open(self.img_info_path, 'w'))
            self.f.close()
            self.is_open = False
        else:
            return

    def write_meta(self, vid_idx, id_, meta_data):
        assert self.is_writable
        meta_info = MetaInfo(meta_data=meta_data,
                             id_=id_)
        self.meta_dict[vid_idx].append(meta_info)

    def write(self, vid_idx, id_, image):
        assert self.is_writable
        loc = self.f.tell()
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        pad = 4 - (len(img_str) % 4)
        record = img_str.ljust(len(img_str) + pad, b'\0')
        img_info = ImgInfo(loc=loc,
                           length=len(record),
                           pad=pad)
        self.img_dict[vid_idx].append(img_info)
        self.f.write(record)

    def read(self, img_info):
        assert not self.is_writable
        self.f.seek(img_info.loc)
        record = self.f.read(img_info.length)
        img_str = record[:-img_info.pad]
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def reset(self):
        self.close()
        self.open()

    def seek(self, loc):
        self.f.seek(loc)


class ChunkWriter(object):

    def __init__(self, adapter, output_folder):
        self.adapter = adapter
        self.output_folder = output_folder

    def initialize_filenames(self, output_folder, chunk_no):
        bin_file_path = os.path.join(output_folder,
                                     'data{}.bin'.format(chunk_no))
        meta_file_path = os.path.join(output_folder,
                                      'meta{}.bin'.format(chunk_no))
        img_info_path = os.path.join(output_folder,
                                     'img_info{}.bin'.format(chunk_no))
        return bin_file_path, img_info_path, meta_file_path

    def write_chunk(self, input_chunk, chunk_id):
        (bin_file_path,
         img_info_path,
         meta_file_path) = self.initialize_filenames(self.output_folder,
                                                     chunk_id)
        gulp_file = GulpVideoIO(bin_file_path,
                                'wb',
                                meta_file_path,
                                img_info_path)
        gulp_file.open()
        for video in self.adapter.iter_data(input_chunk[0], input_chunk[1]):
            id_ = video['id']
            meta_information = video['meta']
            frames = video['frames']

            gulp_file.write_meta(chunk_id, id_, meta_information)
            for frame in frames:
                gulp_file.write(chunk_id, id_, frame)

        gulp_file.close()


def calculate_chunks(videos_per_chunk, num_videos):
    quotient, remainder = divmod(num_videos, videos_per_chunk)
    return [(i, min(i + videos_per_chunk, num_videos))
            for i in range(0, num_videos, videos_per_chunk)]


#class ChunkGenerator(object):
#
#    def __init__(self, adapter, videos_per_chunk):
#        self.adapter = adapter
#        self.videos_per_chunk = videos_per_chunk
#        q, r = divmod(len(adapter), videos_per_chunk)
#        self.length = q + (1 if r else 0)
#
#    def __iter__(self):
#        chunk, videos_in_chunk = [], 0
#        for i in self.adapter.iter_data():
#            chunk.append(i)
#            videos_in_chunk += 1
#            if videos_in_chunk == self.videos_per_chunk:
#                yield chunk
#                chunk, videos_in_chunk = [], 0
#        else:
#            # If the last chunk has less videos than 'videos_per_chunk', we
#            # need to yield it too.
#            if chunk:
#                yield chunk
#
#    def __len__(self):
#        return self.length


class GulpIngestor(object):

    def __init__(self, adapter, output_folder, videos_per_chunk, num_workers):
        self.adapter = adapter
        self.output_folder = output_folder
        self.videos_per_chunk = videos_per_chunk
        self.num_workers = num_workers

    def ingest(self):
        ensure_output_dir_exists(self.output_folder)
        chunks = calculate_chunks(self.videos_per_chunk, len(self.adapter))
        chunk_writer = ChunkWriter(self.adapter, self.output_folder)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            result = executor.map(chunk_writer.write_chunk, chunks,
                                  range(len(chunks)), chunksize=1)
            for r in tqdm(result, total=len(chunks)):
                pass


# class FramesGenerator(object):
#
#     def __init__(self, filenames, id_):
#         self.filenames = filenames
#         self.id_ = id_
#         self.input_type = self.detect_input(self.filenames)
#
#     def extract_frames(self, img_size, shm_dir_path=''):
#         if self.input_type == 'missing':
#             print("no input files given for {}".format(self.id_))
#         if self.input_type == 'video':
#             self. filenames = self.burst_video_into_frames(self.filenames[0],
#                                                            shm_dir_path)
#         return self.get_resized_image(self.filenames, img_size)
#
#     def detect_input(self, input_files):
#         if len(input_files) == 0:
#             return 'missing'
#         if ('.mp4' in input_files[0] and len(input_files) == 1):
#             # TODO improvable
#             return 'video'
#         elif all(('jpg' in i or 'png' in i) for i in input_files):
#             # TODO improvable... Only jpg?
#             return 'frames'
