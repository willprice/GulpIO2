#!/usr/bin/env python

import os
import re
import pickle
import json
import glob
import numpy as np

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from collections import namedtuple, OrderedDict

from tqdm import tqdm

from .utils import img_to_jpeg_bytes, jpeg_bytes_to_img

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

    Parameters
    ----------
    output_dir: str
        Path to the directory containing the files.
    jpeg_decoder: callable that takes a JPEG stored as :py:class:`bytes` and returns
        the desired decoded image format (e.g. np.ndarray)

    Attributes
    ----------
    all_meta_dicts: list of dicts
        All meta dicts from all chunks as a list.
    chunk_lookup: dict: int -> str
        Mapping element id to chunk index.
    chunk_objs_lookup: dict: int -> GulpChunk
        Mapping element id to chunk index.
    merged_meta_dict: dict: id -> meta dict
        all meta dicts merged

    """

    def __init__(self, output_dir, jpeg_decoder=jpeg_bytes_to_img):
        self.output_dir = output_dir
        self.jpeg_decoder = jpeg_decoder
        self.chunk_objs_lookup = OrderedDict(zip(self._chunk_ids(), self._chunks()))
        self.all_meta_dicts = [c.meta_dict for c in self.chunk_objs_lookup.values()]
        self.num_chunks = len(self.chunk_objs_lookup)
        self.chunk_lookup = {}
        for chunk_id, chunk in self.chunk_objs_lookup.items():
            for id_ in chunk.meta_dict:
                self.chunk_lookup[id_] = chunk_id
        self.merged_meta_dict = {}
        for d in self.all_meta_dicts:
            for k in d.keys():
                assert k not in self.merged_meta_dict,\
                    "Duplicate id detected {}".format(k)
            else:
                self.merged_meta_dict.update(d)

    def __iter__(self):
        return iter(self.chunk_objs_lookup.values())

    def chunks(self):
        """ Return a generator over existing GulpChunk objects which are ready
        to be opened and read from. """
        return self.__iter__()

    def _chunks(self):
        return (GulpChunk(*paths, jpeg_decoder=self.jpeg_decoder) for paths in
                self._existing_file_paths())

    def new_chunks(self, total_new_chunks):
        """ Return a generator over freshly setup GulpChunk objects which are ready
        to be opened and written to.

        Parameters
        ----------
        total_new_chunks: int
            The total number of new chunks to initialize.
        """
        return ((GulpChunk(*paths, jpeg_decoder=self.jpeg_decoder) for paths in
                 self._allocate_new_file_paths(total_new_chunks)))

    def __getitem__(self, element):
        id_, _ = extract_input_for_getitem(element)
        chunk_id = self.chunk_lookup[id_]
        gulp_chunk = self.chunk_objs_lookup[chunk_id]
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
        return [int(re.findall(r'\d+', os.path.basename(p))[0]) for p in paths]

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
    """ Represents a gulp chunk on disk.

    Parameters
    ----------
    data_file_path: str
        Path to the *.gulp file.
    meta_file_path: str
        Path to the *.gmeta file.
    serializer: subclass of AbstractSerializer
        The type of serializer to use.
    jpeg_decoder: callable that takes a JPEG stored as :py:class:`bytes` and returns
        the desired decoded image format (e.g. np.ndarray)
    """

    def __init__(self, data_file_path, meta_file_path,
                 serializer=json_serializer, jpeg_decoder=jpeg_bytes_to_img):
        self.jpeg_decoder = jpeg_decoder
        self.serializer = serializer
        self.data_file_path = data_file_path
        self.meta_file_path = meta_file_path
        self.meta_dict = self._get_or_create_dict()

        self._img_info = {}
        self.fp = None

    def __contains__(self, id_):
        return str(id_) in self.meta_dict

    def __getitem__(self, element):
        id_, slice_ = extract_input_for_getitem(element)
        return self.read_frames(id_, slice_)

    def __iter__(self):
        return self.iter_all()

    def _get_frame_infos(self, id_):
        id_ = str(id_)
        if id_ in self.meta_dict:
            return (self._get_or_create_img_info(id_),
                    self._copy_meta_data(id_))

    def _copy_meta_data(self, id_):
        return dict(self.meta_dict[id_]['meta_data'][0])

    def _get_or_create_img_info(self, id_):
        if id_ not in self._img_info:
            self._img_info[id_] = [ImgInfo(*info) for info in self.meta_dict[id_]['frame_info']]
        return self._img_info[id_]

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

    def _append_meta(self, id_, meta_data):
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['meta_data'].append(meta_data)

    def _write_frame(self, id_, image):
        loc = self.fp.tell()
        img_str = img_to_jpeg_bytes(image)
        assert len(img_str) > 0
        pad = self._pad_image(len(img_str))
        record = img_str.ljust(len(img_str) + pad, b'\0')
        assert len(record) > 0
        img_info = ImgInfo(loc=loc,
                           length=len(record),
                           pad=pad)
        id_ = str(id_)
        if id_ not in self.meta_dict:  # implements an OrderedDefaultDict
            self.meta_dict[id_] = self._default_factory()
        self.meta_dict[id_]['frame_info'].append(img_info)
        self.fp.write(record)

    def _write_frames(self, id_, frames):
        for frame in frames:
            self._write_frame(id_, frame)

    @contextmanager
    def open(self, flag='rb'):
        """Open the gulp chunk for reading.

        Parameters
        ----------
        flag: str
            'rb': Read binary
            'wb': Write binary
            'ab': Append to binary

        Notes
        -----
        Works as a context manager but returns None.

        """
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
        """Flush all buffers and write the meta file."""
        self.fp.flush()
        self.serializer.dump(self.meta_dict, self.meta_file_path)

    def append(self, id_, meta_data, frames):
        """ Append an item to the gulp.

        Parameters
        ----------
        id_ : str
            The ID of the item
        meta_data: dict
            The meta-data associated with the item.
        frames: list of numpy arrays
            The frames of the item as a list of numpy dictionaries consisting
            of image pixel values.

        """
        self._append_meta(id_, meta_data)
        self._write_frames(id_, frames)

    def read_frames(self, id_, slice_=None):
        """ Read frames for a single item.

        Parameters
        ----------
        id_: str
            The ID of the item
        slice_: slice or list of ints:
            A slice or list of indices with which to select frames.

        Returns
        -------
        frames (int), meta(dict)
            The frames of the item as a list of numpy arrays consisting of
            image pixel values. And the metadata.

        """
        frame_infos, meta_data = self._get_frame_infos(id_)
        slice_element = slice_ if slice_ is not None else slice(0, len(frame_infos))

        def extract_frame(frame_info):
            self.fp.seek(frame_info.loc)
            record = self.fp.read(frame_info.length)
            img_str = record[:len(record)-frame_info.pad]
            img = self.jpeg_decoder(img_str)
            return img
        if isinstance(slice_element, (list, np.ndarray)):
            selected_frame_infos = [frame_infos[idx] for idx in slice_element]
        else:
            selected_frame_infos = frame_infos[slice_element]
        frames = [extract_frame(frame_info)
                  for frame_info in selected_frame_infos]
        return frames, meta_data

    def iter_all(self, accepted_ids=None, shuffle=False):
        """ Iterate over all frames in the gulp.

        Parameters
        ----------
        accepted_ids: list of str
            A filter for accepted ids.
        shuffle: bool
            Shuffle the items or not.

        Returns
        -------
        iterator
            An iterator that yield a series of frames,meta tuples. See
            `read_frames` for details.
        """

        ids = self.meta_dict.keys()

        if accepted_ids is not None:
            intersection = list(set(ids) & set(accepted_ids))
            ids = [id_ for id_ in ids if id_ in intersection]

        if shuffle:
            ids = list(ids)
            np.random.shuffle(ids)

        with self.open('rb'):
            for id_ in ids:
                frames, meta = self.read_frames(id_)
                yield frames, meta


class ChunkWriter(object):
    """Can write from an adapter to a gulp chunk.

    Parameters
    ----------
    adapter: subclass of AbstractDatasetAdapter
       The adapter to get items from.

    """

    def __init__(self, adapter):
        self.adapter = adapter

    def write_chunk(self, output_chunk, input_slice):
        """Write from an input slice in the adapter to an output chunk.

        Parameters
        ----------
        output_chunk: GulpChunk
           The chunk to write to
        input_slice: slice
           The slice to use from the adapter.

        """
        with output_chunk.open('wb'):
            for video in self.adapter.iter_data(input_slice):
                id_ = video['id']
                meta_data = video['meta']
                frames = video['frames']
                if len(frames) > 0:
                    output_chunk.append(id_, meta_data, frames)
                else:
                    print("Failed to write video with id: {}; no frames"
                          .format(id_))


def calculate_chunk_slices(items_per_chunk, num_items):
    """Calculate slices for indexing an adapter.

    Parameters
    ----------
    items_per_chunk: int
        Approximate number of items per chunk.
    num_items: int
        Total number of items.

    Returns
    -------
    list of slices

    """
    assert items_per_chunk > 0
    assert num_items > 0
    return [slice(i, min(i + items_per_chunk, num_items))
            for i in range(0, num_items, items_per_chunk)]


class GulpIngestor(object):
    """Ingest items from an adapter into an gulp chunks.

    Parameters
    ----------
    adapter: subclass of AbstractDatasetAdapter
        The adapter to ingest from.
    output_folder: str
        The folder/directory to write to.
    videos_per_chunk: int
        The total number of items per chunk.
    num_workers: int
        The level of parallelism.

    """
    def __init__(self, adapter, output_folder, videos_per_chunk, num_workers):
        assert int(num_workers) > 0
        self.adapter = adapter
        self.output_folder = output_folder
        self.videos_per_chunk = int(videos_per_chunk)
        self.num_workers = int(num_workers)

    def __call__(self):
        os.makedirs(self.output_folder, exist_ok=True)
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
