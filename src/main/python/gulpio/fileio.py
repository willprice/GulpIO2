#!/usr/bin/env python

import os
import cv2
import pickle
import json
import glob
import numpy as np

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from PIL import Image
from collections import namedtuple, OrderedDict
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


def extract_input_for_getitem(element):
    if isinstance(element, tuple) and len(element) == 2:
        id_, slice_ = element
    elif isinstance(element, (int, str)):
        id_, slice_ = element, None
    else:
        raise TypeError("Undefined input type! id or (id, slice) expected")
    id_ = str(id_)
    return id_, slice_


class GulpDirectory(object):
    """ Represents a directory containing *.gulp and *.gmeta files.

    Args:
        output_dir: (str) path to the directory containing the files.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.all_meta_dicts = [c.meta_dict for c in self.chunks()]
        self.chunk_lookup = {}
        for chunk_id, meta_dict in zip(self._chunk_ids(), self.all_meta_dicts):
            for id_ in meta_dict:
                self.chunk_lookup[id_] = chunk_id
        self.merged_meta_dict = {}
        for d in self.all_meta_dicts:
            for k in d.keys():
                assert k not in self.merged_meta_dict, "Duplicate id detected {}".format(k)
            else:
                self.merged_meta_dict.update(d)

    def chunks(self):
        """ Return a generator over existing GulpChunk objects which are ready
        to be opened and read from. """
        return ((GulpChunk(*paths) for paths in self._existing_file_paths()))

    def new_chunks(self, total_new_chunks):
        """ Return a generator over freshly setup GulpChunk objects which are ready
        to be opened and written to.

        Args:
            total_new_chunks: (int) the total number of new chunks to setup
        """
        return ((GulpChunk(*paths) for paths in
                 self._allocate_new_file_paths(total_new_chunks)))

    def __getitem__(self, element):
        id_, _ = extract_input_for_getitem(element)
        chunk_id = self.chunk_lookup[id_]
        gulp_chunk = GulpChunk(*self._initialize_filenames(chunk_id))
        with gulp_chunk.open():
            return gulp_chunk[element]

    def _find_existing_data_paths(self):
        return sorted(glob.glob(os.path.join(self.output_dir, 'data*.gulp')))

    def _find_existing_meta_paths(self):
        return sorted(glob.glob(os.path.join(self.output_dir, 'meta*.gmeta')))

    def _load_label_dict(self):
        return json.load(open(os.path.join(self.output_dir, 'label2idx.json'),
                              'rb'))

    def _existing_file_paths(self):
        data_paths = self._find_existing_data_paths()
        meta_paths = self._find_existing_meta_paths()
        assert len(data_paths) == len(meta_paths)
        return zip(data_paths, meta_paths)

    def _find_ids_from_paths(self, paths):
        return [int(p.split('_')[-1].split('.')[0]) for p in paths]

    def _chunk_ids(self):
        data_paths = self._find_existing_data_paths()
        meta_paths = self._find_existing_meta_paths()
        data_ids = self._find_ids_from_paths(data_paths)
        meta_ids = self._find_ids_from_paths(meta_paths)
        assert data_ids == meta_ids
        return data_ids

    def _next_chunk_id(self):
        existing_chunk_ids = self._chunk_ids()
        next_chunk_id = 0
        if len(existing_chunk_ids) > 0:
            next_chunk_id = max([int(i) for i in existing_chunk_ids]) + 1
        return next_chunk_id

    def _allocate_new_file_paths(self, total_new_chunks):
        next_chunk_id = self._next_chunk_id()
        return [self._initialize_filenames(i)
                for i in range(next_chunk_id,
                               next_chunk_id + total_new_chunks)]

    def _initialize_filenames(self, chunk_id):
        data_file_path = os.path.join(
            self.output_dir, 'data_{}.gulp'.format(chunk_id))
        meta_file_path = os.path.join(
            self.output_dir, 'meta_{}.gmeta'.format(chunk_id))
        return data_file_path, meta_file_path


class GulpChunk(object):

    def __init__(self, data_file_path, meta_file_path,
                 serializer=json_serializer):
        self.serializer = serializer
        self.data_file_path = data_file_path
        self.meta_file_path = meta_file_path
        self.meta_dict = self._get_or_create_dict()
        self.fp = None

    def __contains__(self, id_):
        return self._get_frame_infos(id_)

    def __getitem__(self, element):
        id_, slice_ = extract_input_for_getitem(element)
        return self.read_frames(id_, slice_)

    def __iter__(self):
        return self.iter_all()

    def _get_frame_infos(self, id_):
        id_ = str(id_)
        if id_ in self.meta_dict:
            return ([ImgInfo(*info)
                     for info in self.meta_dict[id_]['frame_info']],
                    dict(self.meta_dict[id_]['meta_data'][0]))

    def _get_or_create_dict(self):
        if os.path.exists(self.meta_file_path):
            return self.serializer.load(self.meta_file_path)
        else:
            return OrderedDict()

    @staticmethod
    def _default_factory():
        return OrderedDict([('frame_info', []), ('meta_data', [])])

    @staticmethod
    def _pad_image(number):
        return (4 - (number % 4)) % 4

    def append_meta(self, id_, meta_data):
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['meta_data'].append(meta_data)

    def write_frame(self, id_, image):
        loc = self.fp.tell()
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        pad = self._pad_image(len(img_str))
        record = img_str.ljust(len(img_str) + pad, b'\0')
        img_info = ImgInfo(loc=loc,
                           length=len(record),
                           pad=pad)
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['frame_info'].append(img_info)
        self.fp.write(record)

    def write_frames(self, id_, frames):
        for frame in frames:
            self.write_frame(id_, frame)

    @contextmanager
    def open(self, flag='rb'):
        if flag in ['wb', 'rb', 'ab']:
            self.fp = open(self.data_file_path, flag)
        else:
            m = "This file does not support the mode: '{}'".format(flag)
            raise NotImplementedError(m)
        yield
        if flag in ['wb', 'ab']:
            self.flush()
        self.fp.close()

    def flush(self):
        self.fp.flush()
        self.serializer.dump(self.meta_dict, self.meta_file_path)

    def append(self, id_, meta_data, frames):
        self.append_meta(id_, meta_data)
        self.write_frames(id_, frames)

    def read_frames(self, id_, slice_=None):
        frame_infos, meta_data = self._get_frame_infos(id_)
        frames = []
        slice_element = slice_ or slice(0, len(frame_infos))

        def extract_frame(frame_info):
            self.fp.seek(frame_info.loc)
            record = self.fp.read(frame_info.length)
            img_str = record[:len(record)-frame_info.pad]
            nparr = np.fromstring(img_str, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        frames = [Image.fromarray(extract_frame(frame_info))
                  for frame_info in frame_infos[slice_element]]
        return frames, meta_data

    def iter_all(self, accepted_ids=None, shuffle=False):
        ids = self.meta_dict.keys()

        if accepted_ids is not None:
            intersection = list(set(ids) & set(accepted_ids))
            ids = [id_ for id_ in ids if id_ in intersection]

        if shuffle:
            ids = list(ids)
            np.random.shuffle(ids)

        for id_ in ids:
            frames, meta = self.read_frames(id_)
            yield frames, meta


class ChunkWriter(object):

    def __init__(self, adapter):
        self.adapter = adapter

    def write_chunk(self, input_chunk, input_slice):
        with input_chunk.open('wb'):
            for video in self.adapter.iter_data(input_slice):
                id_ = video['id']
                meta_data = video['meta']
                frames = video['frames']
                if len(frames) > 0:
                    input_chunk.append(id_, meta_data, frames)
                else:
                    print("Failed to write video with id: {}; no frames"
                          .format(id_))


def calculate_chunk_slices(videos_per_chunk, num_videos):
    assert videos_per_chunk > 0
    assert num_videos > 0
    return [slice(i, min(i + videos_per_chunk, num_videos))
            for i in range(0, num_videos, videos_per_chunk)]


class GulpIngestor(object):

    def __init__(self, adapter, output_folder, videos_per_chunk, num_workers):
        assert int(num_workers) > 0
        self.adapter = adapter
        self.output_folder = output_folder
        self.videos_per_chunk = int(videos_per_chunk)
        self.num_workers = int(num_workers)

    def __call__(self):
        ensure_output_dir_exists(self.output_folder)
        chunk_slices = calculate_chunk_slices(self.videos_per_chunk,
                                              len(self.adapter))
        gulp_directory = GulpDirectory(self.output_folder)
        new_chunks = gulp_directory.new_chunks(len(chunk_slices))
        chunk_writer = ChunkWriter(self.adapter)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            result = executor.map(chunk_writer.write_chunk,
                                  new_chunks,
                                  chunk_slices)
            for r in tqdm(result,
                          desc='Chunks finished',
                          unit='chunk',
                          dynamic_ncols=True,
                          total=len(chunk_slices)):
                pass
