import os
import tempfile
import shutil
import json
import pickle
import PIL.Image

from collections import OrderedDict
from io import BytesIO

import numpy as np
import numpy.testing as npt

import unittest
import unittest.mock as mock

from gulpio2.fileio import (GulpChunk,
                           ChunkWriter,
                           GulpIngestor,
                           GulpDirectory,
                           calculate_chunk_slices,
                           json_serializer,
                           pickle_serializer,
                           extract_input_for_getitem,
                           ImgInfo,
                           )
from gulpio2.adapters import AbstractDatasetAdapter


def create_image(shape, val=1, dtype=np.uint8):
    return PIL.Image.fromarray(np.ones(shape, dtype=dtype) * val)


class FSBase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='fileio-test-')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestJSONSerializer(FSBase):

    def test_dump(self):
        filename = os.path.join(self.temp_dir, 'ANY_JSON.json')
        content = {'ANY_KEY': 'ANY_CONTENT'}
        json_serializer.dump(content, filename)
        with open(filename, 'r') as fp:
            written_content = json.load(fp)
        self.assertEqual(content, written_content)

    def test_load(self):
        filename = os.path.join(self.temp_dir, 'ANY_JSON.json')
        content = {'ANY_KEY': 'ANY_CONTENT'}
        json_serializer.dump(content, filename)
        received_content = json_serializer.load(filename)
        self.assertEqual(content, received_content)


class TestPickleSerializer(FSBase):

    def test_dump(self):
        filename = os.path.join(self.temp_dir, 'ANY_PICKLE.pickle')
        content = {'ANY_KEY': 'ANY_CONTENT'}
        pickle_serializer.dump(content, filename)
        with open(filename, 'rb') as file_pointer:
            written_content = pickle.load(file_pointer)
        self.assertEqual(content, written_content)

    def test_load(self):
        filename = os.path.join(self.temp_dir, 'ANY_PICKLE.pickle')
        content = {'ANY_KEY': 'ANY_CONTENT'}
        pickle_serializer.dump(content, filename)
        received_content = pickle_serializer.load(filename)
        self.assertEqual(content, received_content)


class TestExtractInputForGetitem(unittest.TestCase):

    def test_input_int(self):
        id_ = 1
        res_id, res_slice = extract_input_for_getitem(id_)
        self.assertEqual(res_id, "1")
        self.assertEqual(res_slice, None)

    def test_input_tuple(self):
        element = (1, slice(1, 2))
        res_id, res_slice = extract_input_for_getitem(element)
        self.assertEqual(res_id, "1")
        self.assertEqual(res_slice, slice(1, 2))

    def test_input_too_large_tuple(self):
        element = (1, 2, 3)
        with self.assertRaises(TypeError):
            extract_input_for_getitem(element)

    def test_wrong_input_type(self):
        element = 0.1
        with self.assertRaises(TypeError):
            extract_input_for_getitem(element)


class TestCalculateChunks(unittest.TestCase):

    def test_one_video_in_chunk(self):
        expected = [slice(0, 1), slice(1, 2)]
        result = calculate_chunk_slices(1, 2)
        self.assertEqual(expected, result)

    def test_two_items_in_chunk_last_chunk_not_full(self):
        expected = [slice(0, 2), slice(2, 3)]
        result = calculate_chunk_slices(2, 3)
        self.assertEqual(expected, result)

    def test_two_items_in_chunk_last_chunk_full(self):
        expected = [slice(0, 2), slice(2, 4)]
        result = calculate_chunk_slices(2, 4)
        self.assertEqual(expected, result)

    def test_items_per_chunk_larger_num_items(self):
        expected = [slice(0, 2)]
        result = calculate_chunk_slices(100, 2)
        self.assertEqual(expected, result)

    def test_no_items_in_chunk(self):
        self.assertRaises(AssertionError, calculate_chunk_slices, 0, 1)

    def test_num_items_is_zero(self):
        self.assertRaises(AssertionError, calculate_chunk_slices, 1, 0)


class GulpChunkElement(FSBase):

    @mock.patch('gulpio2.fileio.json_serializer')
    def setUp(self, mock_json_serializer):
        super().setUp()
        self.mock_json_serializer = mock_json_serializer
        self.data_file_path = os.path.join(self.temp_dir, 'ANY_DATA_FILE_PATH')
        self.meta_file_path = os.path.join(self.temp_dir, 'ANY_META_FILE_PATH')
        self.gulp_chunk = GulpChunk(self.data_file_path,
                                    self.meta_file_path,
                                    mock_json_serializer)


class TestGulpChunk(GulpChunkElement):

    def test_initializer(self):
        self.assertEqual(self.data_file_path,
                         self.gulp_chunk.data_file_path)
        self.assertEqual(self.meta_file_path,
                         self.gulp_chunk.meta_file_path)
        self.assertEqual(self.mock_json_serializer,
                         self.gulp_chunk.serializer)

        self.assertEqual(self.gulp_chunk.meta_dict, OrderedDict())
        self.assertEqual(self.gulp_chunk.fp, None)

    def test_get_or_create_dict_not_exists(self):
        self.assertEqual(self.gulp_chunk._get_or_create_dict(), OrderedDict())

    def test_get_or_create_dict_exists(self):
        open(self.meta_file_path, 'w').close()
        self.gulp_chunk._get_or_create_dict()
        self.mock_json_serializer.load.called_once_with(self.meta_file_path)

    def test_default_factory(self):
        expected = OrderedDict([('frame_info', []), ('meta_data', [])])
        self.assertEqual(expected, self.gulp_chunk._default_factory())

    def test_open_with_wb(self):
        self.gulp_chunk.flush = mock.Mock()
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            with self.gulp_chunk.open('wb'):
                m.assert_called_once_with(
                    self.gulp_chunk.data_file_path, 'wb')
            self.gulp_chunk.flush.assert_called_once_with()

    def test_open_with_rb(self):
        self.gulp_chunk.flush = mock.Mock()
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            with self.gulp_chunk.open('rb'):
                m.assert_called_once_with(
                    self.gulp_chunk.data_file_path, 'rb')
            self.assertFalse(self.gulp_chunk.flush.called)

    def test_open_with_ab(self):
        self.gulp_chunk.flush = mock.Mock()
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            with self.gulp_chunk.open('ab'):
                m.assert_called_once_with(
                    self.gulp_chunk.data_file_path, 'ab')
            self.gulp_chunk.flush.assert_called_once_with()

    def test_open_unknown_flag(self):
        get_mock = mock.Mock()
        self.gulp_chunk._get_or_create_dict = get_mock
        with self.assertRaises(NotImplementedError):
            with self.gulp_chunk.open('NO_SUCH_FLAG'):
                pass

    def test_flush(self):
        self.gulp_chunk.serializer = json_serializer
        self.gulp_chunk.meta_dict = {'0': {'meta_data': []}}
        self.gulp_chunk.fp = mock.Mock()
        self.gulp_chunk.flush()
        meta_path_written = open(self.meta_file_path).read()
        self.assertEqual('{"0": {"meta_data": []}}', meta_path_written)
        self.gulp_chunk.fp.flush.assert_called_once_with()

    def test_append_meta(self):
        self.gulp_chunk.meta_dict = {'0': {'meta_data': []}}
        self.gulp_chunk._append_meta(0, {'meta': 'ANY_META'})
        expected = {'0': {'meta_data': [{'meta': 'ANY_META'}]}}
        self.assertEqual(expected, self.gulp_chunk.meta_dict)

    def test_append_meta_initializes_correctly(self):
        self.gulp_chunk.meta_dict = {}
        self.gulp_chunk._append_meta(0, {'meta': 'ANY_META'})
        expected = {'0': {'meta_data': [{'meta': 'ANY_META'}],
                          'frame_info': []}}
        self.assertEqual(expected, self.gulp_chunk.meta_dict)

    def test_pad_image(self):
        self.assertEqual(0, GulpChunk._pad_image(0))
        self.assertEqual(1, GulpChunk._pad_image(3))
        self.assertEqual(2, GulpChunk._pad_image(2))
        self.assertEqual(3, GulpChunk._pad_image(1))
        self.assertEqual(0, GulpChunk._pad_image(4))

    def test_write_frame(self):
        bio = BytesIO()
        self.gulp_chunk.meta_dict = {'0': {'meta_data': [{'test': 'ANY'}],
                                           'frame_info': [[1, 2, 3]]}}
        self.gulp_chunk.fp = bio
        with mock.patch('gulpio2.fileio.img_to_jpeg_bytes') as img_to_jpeg_bytes_mock:
            img_to_jpeg_bytes_mock.return_value = np.ones(1, np.uint8).tobytes()
            self.gulp_chunk._write_frame(0, None)
            self.assertEqual(b'\x01\x00\x00\x00', bio.getvalue())
            expected = {'0': {'meta_data': [{'test': 'ANY'}],
                              'frame_info': [[1, 2, 3], ImgInfo(0, 3, 4)]}}
            self.assertEqual(expected, self.gulp_chunk.meta_dict)

    def test_write_frame_new_entry(self):
        bio = BytesIO()
        self.gulp_chunk.meta_dict = {}
        self.gulp_chunk.fp = bio
        with mock.patch('gulpio2.fileio.img_to_jpeg_bytes') as img_to_jpeg_bytes_mock:
            img_to_jpeg_bytes_mock.return_value = np.ones((1,), dtype='uint8').tobytes()
            self.gulp_chunk._write_frame(0, None)
            self.assertEqual(b'\x01\x00\x00\x00', bio.getvalue())
            expected = {'0': {'meta_data': [],
                              'frame_info': [ImgInfo(0, 3, 4)]}}
            self.assertEqual(expected, self.gulp_chunk.meta_dict)

    def test_get_frame_infos(self):
        self.gulp_chunk.meta_dict = {'0': {'meta_data': [{'meta': 'ANY_META'}],
                                           'frame_info': [[1, 2, 3]]}}
        with mock.patch('gulpio2.fileio.GulpChunk.open'):
            output = self.gulp_chunk._get_frame_infos('0')
        expected = ([ImgInfo(loc=1, pad=2, length=3)], {'meta': 'ANY_META'})
        self.assertEqual(expected, output)

    def test_contains(self):
        self.gulp_chunk.meta_dict = {'0': {'meta_data': [{}],
                                           'frame_info': []}}
        with mock.patch('gulpio2.fileio.GulpChunk.open'):
            output = 0 in self.gulp_chunk
        self.assertTrue(output)

    def test_contains_if_id_not_in_chunk(self):
        self.gulp_chunk.meta_dict = {'0': {'meta_data': [{}],
                                           'frame_info': []}}
        with mock.patch('gulpio2.fileio.GulpChunk.open'):
            output = 1 in self.gulp_chunk
        self.assertFalse(output)

    def test_read_frames(self):
        # use 'write_frame' to write a single image
        self.gulp_chunk.meta_dict = OrderedDict()
        self.gulp_chunk.fp = BytesIO()
        image = create_image((3, 3, 3))
        self.gulp_chunk._write_frame(0, image)
        self.gulp_chunk.meta_dict['0']['meta_data'].append({})

        # recover the single frame using 'read'
        frames, meta = self.gulp_chunk.read_frames('0')
        npt.assert_array_equal(np.asarray(image), np.array(frames[0]))
        self.assertEqual({}, meta)

    def test_read_frames_fixed_length(self):
        # use 'write_frame' to write a single image
        self.gulp_chunk.meta_dict = OrderedDict()
        self.gulp_chunk.fp = BytesIO()
        image = create_image((1, 4))
        self.gulp_chunk._write_frame(0, image)
        self.gulp_chunk.meta_dict['0']['meta_data'].append({})
        # recover the single frame using 'read'
        frames, meta = self.gulp_chunk.read_frames('0')
        npt.assert_array_equal(image, np.array(frames[0]))
        self.assertEqual({}, meta)

    def test_iter(self):
        read_mock = mock.Mock()
        read_mock.return_value = [], []
        self.gulp_chunk.meta_dict = OrderedDict((('0', {}),
                                                 ('1', {}),
                                                 ('2', {}),
                                                 ('3', {}),
                                                 ('4', {})))
        self.gulp_chunk.read_frames = read_mock
        open_mock = mock.MagicMock()
        self.gulp_chunk.open = open_mock

        [_ for _ in self.gulp_chunk]
        read_mock.assert_has_calls([mock.call('0'),
                                    mock.call('1'),
                                    mock.call('2'),
                                    mock.call('3'),
                                    mock.call('4')])

    def test_iter_all(self):
        read_mock = mock.Mock()
        read_mock.return_value = [], []
        self.gulp_chunk.meta_dict = OrderedDict((('0', {}),
                                                 ('1', {}),
                                                 ('2', {}),
                                                 ('3', {}),
                                                 ('4', {})))
        self.gulp_chunk.read_frames = read_mock
        self.gulp_chunk.read_frames = read_mock
        open_mock = mock.MagicMock()
        self.gulp_chunk.open = open_mock

        [_ for _ in self.gulp_chunk.iter_all()]
        read_mock.assert_has_calls([mock.call('0'),
                                    mock.call('1'),
                                    mock.call('2'),
                                    mock.call('3'),
                                    mock.call('4')])

        # test with filtering
        [_ for _ in self.gulp_chunk.iter_all(accepted_ids=['0', '1', '2'])]
        read_mock.assert_has_calls([mock.call('0'),
                                    mock.call('1'),
                                    mock.call('2')])

        # test with shuffling
        with mock.patch('numpy.random.shuffle') as shuffle_mock:
            np.random.seed(123)
            [_ for _ in self.gulp_chunk.iter_all(shuffle=True)]
            # check numpy.random.shuffle has been called
            ids = ['{}'.format(i) for i in range(5)]
            shuffle_mock.assert_called_once_with(ids)
            # check videos are accessed in the correct order
            np.random.seed(123)
            np.random.shuffle(ids)
            read_mock.assert_has_calls([mock.call(id_) for id_ in ids])

        # test with shuffling and filtering
        with mock.patch('numpy.random.shuffle') as shuffle_mock:
            np.random.seed(123)
            [_ for _ in self.gulp_chunk.iter_all(accepted_ids=['0', '1', '2'],
                                                 shuffle=True)]
            # check numpy.random.shuffle has been called
            ids = ['{}'.format(i) for i in range(3)]
            shuffle_mock.assert_called_once_with(ids)
            # check videos are accessed in the correct order
            np.random.seed(123)
            np.random.shuffle(ids)
            read_mock.assert_has_calls([mock.call(id_) for id_ in ids])


class ChunkWriterElement(FSBase):

    def setUp(self):
        super().setUp()
        self.adapter = mock.MagicMock()
        self.adapter.__len__.return_value = 1
        self.output_folder = os.path.join(self.temp_dir, 'ANY_OUTPUT_FOLDER')
        self.chunk_writer = ChunkWriter(self.adapter)


class TestChunkWriter(ChunkWriterElement):

    def test_initialization(self):
        self.assertEqual(self.adapter,
                         self.chunk_writer.adapter)

    @mock.patch('gulpio2.fileio.GulpChunk')
    def test_write_chunk(self, mock_gulp):
        def mock_iter_data(input_slice):
            yield {'id': 0,
                   'meta': {'meta': 'ANY_META'},
                   'frames': ['ANY_FRAME1', 'ANY_FRAME2'],
                   }
        self.adapter.iter_data = mock_iter_data
        self.chunk_writer.write_chunk(mock_gulp, slice(0, 1))
        mock_gulp.append.assert_called_once_with(
            0, {'meta': 'ANY_META'}, ['ANY_FRAME1', 'ANY_FRAME2'])


class GulpIngestorElement(FSBase):

    @mock.patch('gulpio2.adapters.AbstractDatasetAdapter')
    def setUp(self, mock_adapter):
        super().setUp()
        self.adapter = mock_adapter
        self.output_folder = os.path.join(self.temp_dir, 'ANY_OUTPUT_FOLDER')
        self.videos_per_chunk = 1
        self.num_workers = 1
        self.gulp_ingestor = GulpIngestor(self.adapter,
                                          self.output_folder,
                                          self.videos_per_chunk,
                                          self.num_workers)


class TestGulpIngestor(GulpIngestorElement):

    def test_initialization(self):
        self.assertEqual(self.adapter, self.gulp_ingestor.adapter)
        self.assertEqual(self.output_folder, self.gulp_ingestor.output_folder)
        self.assertEqual(self.videos_per_chunk,
                         self.gulp_ingestor.videos_per_chunk)
        self.assertEqual(self.num_workers, self.gulp_ingestor.num_workers)

    @mock.patch('gulpio2.fileio.ChunkWriter')
    @mock.patch('gulpio2.fileio.ProcessPoolExecutor')
    def test_ingest(self,
                    mock_process_pool,
                    mock_chunk_writer,
                    ):

        # The next three lines mock the ProcessPoolExecutor and it's map
        # function.
        executor_mock = mock.Mock()
        executor_mock.map.return_value = []
        mock_process_pool.return_value.__enter__.return_value = executor_mock
        self.gulp_ingestor.adapter.__len__.return_value = 2
        self.gulp_ingestor()
        mock_chunk_writer.assert_called_once_with(self.adapter)

        executor_mock.map.assert_called_once_with(
            mock_chunk_writer.return_value.write_chunk,
            mock.ANY,
            [slice(0, 1), slice(1, 2)],
        )


class DummyVideosAdapter(AbstractDatasetAdapter):

    def __init__(self, num_videos, num_frames=1):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.ids = [str(i) for i in range(num_videos)]
        np.random.shuffle(self.ids)

    def __len__(self):
        return self.num_videos

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for id_ in self.ids[slice_element]:
            yield {
                'meta': {'id': id_},
                'frames': [
                    PIL.Image.fromarray(np.ones((1, 1, 3), dtype=np.uint8) * i)
                    for i in range(self.num_frames)
                ],
                'id': id_,
            }


class RoundTripAdapter(AbstractDatasetAdapter):

    def __init__(self, ids=[0, 1, 2]):
        self.result1 = {
            'meta': {'name': 'empty video'},
            'frames': [],
            'id': ids[0],
        }
        self.result2 = {
            'meta': {'name': 'bunch of numpy arrays'},
            'frames': [
                create_image((4, 1, 3)),
                create_image((3, 1, 3)),
                create_image((2, 1, 3)),
                create_image((1, 1, 3)),
            ],
            'id': ids[1],
        }
        self.result3 = {
            'meta': {'name': 'shorter_video'},
            'frames': [
                create_image((4, 1, 3)),
                create_image((3, 1, 3)),
            ],
            'id': ids[2],
        }
        self.results = [
            self.result1,
            self.result2,
            self.result3,
        ]

    def __len__(self):
        return 3

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for r in self.results[slice_element]:
            yield r


class TestGulpDirectory(FSBase):

    def test_init(self):
        adapter = RoundTripAdapter()
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()
        gulp_directory = GulpDirectory(output_directory)
        self.assertEqual(gulp_directory.output_dir, output_directory)

        expected_all_meta_dicts = [
            OrderedDict([('1',
                         OrderedDict([('frame_info',
                                      [[0, 1, 632],
                                       [632, 1, 632],
                                       [1264, 1, 632],
                                       [1896, 1, 632]]),
                                      ('meta_data',
                                       [OrderedDict(
                                           [('name',
                                             'bunch of numpy arrays')])])]))]),
            OrderedDict([('2',
                        OrderedDict([('frame_info',
                                     [[0, 1, 632],
                                      [632, 1, 632]]),
                                    ('meta_data',
                                     [OrderedDict(
                                        [('name', 'shorter_video')])])]))])]
        self.assertEqual(gulp_directory.all_meta_dicts,
                         expected_all_meta_dicts)

        self.assertEqual(gulp_directory.chunk_lookup, {'1': 0, '2': 1})

        expected_merged_meta_dict = {
            '1': OrderedDict([('frame_info',
                              [[0, 1, 632],
                               [632, 1, 632],
                               [1264, 1, 632],
                               [1896, 1, 632]]),
                              ('meta_data',
                               [OrderedDict(
                                   [('name',
                                     'bunch of numpy arrays')])])]),
            '2': OrderedDict([('frame_info',
                              [[0, 1, 632],
                               [632, 1, 632]]),
                              ('meta_data',
                               [OrderedDict([('name',
                                              'shorter_video')])])])}
        self.assertEqual(gulp_directory.merged_meta_dict,
                         expected_merged_meta_dict)

    def test_round_trip(self):
        # first, write some garbage in
        adapter = RoundTripAdapter()
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()

        # then, read it and make sure the garbage came back out
        gulp_directory = GulpDirectory(output_directory)
        gulp_chunk = next(gulp_directory.chunks())
        expected_output_shapes = [[(4, 1, 3),
                                   (3, 1, 3),
                                   (2, 1, 3),
                                   (1, 1, 3)]
                                  ]
        expected_meta = [{'name': 'bunch of numpy arrays'}]
        for i, (frames, meta) in enumerate(gulp_chunk):
            self.assertEqual(expected_meta[i], meta)
            self.assertEqual(expected_output_shapes[i],
                             [np.array(f).shape for f in frames])

        # check that random_access works
        expected_frames = [
            create_image((4, 1, 3)),
            create_image((3, 1, 3)),
            create_image((2, 1, 3)),
            create_image((1, 1, 3)),
        ]
        received_frames, received_meta = gulp_directory[1]
        for ef, rf in zip(expected_frames, received_frames):
            npt.assert_array_equal(np.asarray(ef), np.asarray(rf))
        self.assertEqual(expected_meta[0], received_meta)

        # now append/extend the gulps
        GulpIngestor(RoundTripAdapter(ids=[3, 4, 5]),
                     output_directory, 2, 1)()

        # then, read it again, using __iter__
        gulp_directory = GulpDirectory(output_directory)
        gulp_chunk = next(iter(gulp_directory))
        expected_output_shapes = [(4, 1, 3),
                                  (3, 1, 3),
                                  (2, 1, 3),
                                  (1, 1, 3)]
        expected_meta = {'name': 'bunch of numpy arrays'}

        for frames, meta in gulp_chunk:
            self.assertEqual(expected_meta, meta)
            self.assertEqual(expected_output_shapes,
                             [np.array(f).shape for f in frames])

    def test_random_access(self):
        # ingest dummy videos
        adapter = DummyVideosAdapter(num_videos=25)
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()

        # create gulp directory
        gulp_directory = GulpDirectory(output_directory)

        # check all videos can be accessed
        for id_ in adapter.ids:
            with self.subTest(id_=id_):
                # check img id is in the lookup table
                self.assertTrue(id_ in gulp_directory.chunk_lookup)
                # check the img can be accessed
                img, meta = gulp_directory[id_]
                # check the meta id match
                self.assertEqual(meta['id'], id_)

    def test_random_access_with_sparse_frames(self):
        # ingest dummy videos
        adapter = DummyVideosAdapter(num_videos=5, num_frames=10)
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()

        # create gulp directory
        gulp_directory = GulpDirectory(output_directory)

        # check all videos can be accessed
        for id_ in adapter.ids:
            with self.subTest(id_=id_):
                # check img id is in the lookup table
                self.assertTrue(id_ in gulp_directory.chunk_lookup)
                # check the img can be accessed
                frame_idxs = [0, 5, 9]
                frames, meta = gulp_directory[id_, frame_idxs]
                # check the meta id match
                self.assertEqual(meta['id'], id_)
                for frame_idx, frame in zip(frame_idxs, frames):
                    # DummyVideosAdapter yields frames filled with the frame index
                    # as their value
                    assert (np.asarray(frame) == frame_idx).all()

