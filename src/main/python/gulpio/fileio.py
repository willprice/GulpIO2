#!/usr/bin/env python

import os
import cv2
import pickle
import json
import numpy as np

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from PIL import Image
from collections import namedtuple, defaultdict, OrderedDict
from tqdm import tqdm

from .utils import ensure_output_dir_exists


ImgInfo = namedtuple('ImgInfo', ['loc',
                                 'pad',
                                 'length'])


class AbstractSerializer(ABC):  # pragma: no cover

    @abstractmethod
    def load(self, file_name):
        pass

    @abstractmethod
    def dump(self, thing, file_name):
        pass


class PickleSerializer(AbstractSerializer):

    def load(self, file_name):
        with open(file_name, 'rb') as file_pointer:
            return pickle.load(file_pointer)

    def dump(self, thing, file_name):
        with open(file_name, 'wb') as file_pointer:
            pickle.dump(thing, file_pointer)


class JSONSerializer(AbstractSerializer):

    def load(self, file_name):
        with open(file_name, 'r') as file_pointer:
            return json.load(file_pointer, object_pairs_hook=OrderedDict)

    def dump(self, thing, file_name):
        with open(file_name, 'w') as file_pointer:
            json.dump(thing, file_pointer)


pickle_serializer = PickleSerializer()
json_serializer = JSONSerializer()


class GulpChunk(object):

    def __init__(self, chunk_id, output_path, expected_chunks,
                 serializer=json_serializer):
        self.serializer = serializer
        self.output_path = output_path
        self.expected_chunks = expected_chunks
        self.meta_dict = None
        (self.data_file_path,
         self.meta_file_path) = self.initialize_filenames(chunk_id)

    def get_or_create_dict(self, path):
        if os.path.exists(path):
            return self.serializer.load(path)
        return OrderedDict()

    @staticmethod
    def default_factory():
        return {'meta_data': [], 'frame_info': []}

    @contextmanager
    def open(self, flag='rb'):
        self.meta_dict = self.get_or_create_dict(self.meta_file_path)
        if flag == 'wb':
            fp = open(self.data_file_path, flag)
        elif flag == 'rb':
            fp = open(self.data_file_path, flag)
        else:
            m = "This file does not support the mode: '{}'".format(flag)
            raise NotImplementedError(m)
        yield fp
        self.flush()
        fp.close()

    def flush(self):
        self.serializer.dump(self.meta_dict, self.meta_file_path)

    def initialize_filenames(self, chunk_no):
        padded_chunk_no = self.pad_chunk_no(chunk_no)
        bin_file_path = os.path.join(self.output_path,
                                     'data_{}.gulp'.format(padded_chunk_no))
        meta_file_path = os.path.join(self.output_path,
                                      'meta_{}.gmeta'.format(padded_chunk_no))
        return bin_file_path, meta_file_path

    def pad_chunk_no(self, chunk_no):
        return str(chunk_no).zfill(len(str(self.expected_chunks)))

    def append_meta(self, id_, meta_data):
        if str(id_) not in self.meta_dict:
            self.meta_dict[str(id_)] = self.default_factory()
        self.meta_dict[str(id_)]['meta_data'].append(meta_data)

    def write_frame(self, fp, id_, image):
        loc = fp.tell()
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        pad = 4 - (len(img_str) % 4)
        record = img_str.ljust(len(img_str) + pad, b'\0')
        img_info = ImgInfo(loc=loc,
                           length=len(record),
                           pad=pad)
        if str(id_) not in self.meta_dict:
            self.meta_dict[str(id_)] = self.default_factory()
        self.meta_dict[str(id_)]['frame_info'].append(img_info)
        fp.write(record)

    def retrieve_meta_infos(self, id_):
        return ([ImgInfo(*self.meta_dict[str(id_)]['frame_info'][i])
                 for i in range(len(self.meta_dict[id_]['frame_info']))],
                dict(self.meta_dict[str(id_)]['meta_data'][0]))

    def read_frames(self, fp, id_):
        frame_infos, meta_data = self.retrieve_meta_infos(id_)
        frames = []
        for frame_info in frame_infos:
            fp.seek(frame_info.loc)
            record = fp.read(frame_info.length)
            img_str = record[:-frame_info.pad]
            nparr = np.fromstring(img_str, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        return frames, meta_data

    def read_chunk(self, fp):
        for i, id_ in enumerate(self.meta_dict.keys()):
            frames, meta = self.read_frames(fp, id_)
            yield (frames, meta)


class ChunkWriter(object):

    def __init__(self, adapter, output_folder, videos_per_chunk):
        self.adapter = adapter
        self.output_folder = output_folder
        self.videos_per_chunk = videos_per_chunk
        self.chunks = calculate_chunks(self.videos_per_chunk,
                                       len(self.adapter))

    def __len__(self):
        return len(self.chunks)

    def write_chunk(self, input_chunk, chunk_id):
        gulp_file = GulpChunk(chunk_id, self.output_folder, len(self))
        with gulp_file.open('wb') as fp:
            for video in self.adapter.iter_data(slice(*input_chunk)):
                id_ = video['id']
                meta_information = video['meta']
                frames = video['frames']
                if len(frames) > 0:
                    gulp_file.append_meta(id_, meta_information)
                    for frame in frames:
                        gulp_file.write_frame(fp, id_, frame)


def calculate_chunks(videos_per_chunk, num_videos):
    assert videos_per_chunk > 0
    assert num_videos > 0
    return [(i, min(i + videos_per_chunk, num_videos))
            for i in range(0, num_videos, videos_per_chunk)]


class GulpIngestor(object):

    def __init__(self, adapter, output_folder, videos_per_chunk, num_workers):
        assert num_workers > 0
        self.adapter = adapter
        self.output_folder = output_folder
        self.videos_per_chunk = videos_per_chunk
        self.num_workers = num_workers

    def __call__(self):
        ensure_output_dir_exists(self.output_folder)
        chunk_writer = ChunkWriter(self.adapter,
                                   self.output_folder,
                                   self.videos_per_chunk)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            result = executor.map(chunk_writer.write_chunk,
                                  chunk_writer.chunks,
                                  range(len(chunk_writer)))
            for r in tqdm(result,
                          desc='Chunks finished',
                          unit='chunk',
                          dynamic_ncols=True,
                          total=len(chunk_writer)):
                pass
