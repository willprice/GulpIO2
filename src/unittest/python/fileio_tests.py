import os
import tempfile
import shutil
import json
import pickle

from collections import defaultdict
from io import BytesIO

import numpy

import unittest
import unittest.mock as mock

from gulpio.fileio import (GulpChunk,
                           ChunkWriter,
                           GulpIngestor,
                           calculate_chunks,
                           json_serializer,
                           pickle_serializer,
                           ImgInfo,
                           )
from gulpio.adapters import AbstractDatasetAdapter


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


class TestCalculateChunks(unittest.TestCase):

    def test_one_video_in_chunk(self):
        expected = [(0, 1), (1, 2)]
        result = calculate_chunks(1, 2)
        self.assertEqual(expected, result)

    def test_two_videos_in_chunk_last_chunk_not_full(self):
        expected = [(0, 2), (2, 3)]
        result = calculate_chunks(2, 3)
        self.assertEqual(expected, result)

    def test_two_videos_in_chunk_last_chunk_full(self):
        expected = [(0, 2), (2, 4)]
        result = calculate_chunks(2, 4)
        self.assertEqual(expected, result)

    def test_videos_per_chunk_larger_num_videos(self):
        expected = [(0, 2)]
        result = calculate_chunks(100, 2)
        self.assertEqual(expected, result)

    def test_no_videos_in_chunk(self):
        self.assertRaises(AssertionError, calculate_chunks, 0, 1)

    def test_num_videos_is_zero(self):
        self.assertRaises(AssertionError, calculate_chunks, 1, 0)


class GulpChunkElement(FSBase):

    @mock.patch('gulpio.fileio.json_serializer')
    def setUp(self, mock_json_serializer):
        super().setUp()
        self.mock_json_serializer = mock_json_serializer
        self.output_path = "ANY_OUTPUT_PATH"
        self.gulp_video_io = GulpChunk(0,
                                       self.output_path,
                                       1,
                                       mock_json_serializer)

    def tearDown(self):
        super().tearDown()


class TestGulpChunk(GulpChunkElement):

    def test_initializer(self):
        self.assertEqual('ANY_OUTPUT_PATH/data_0.gulp',
                         self.gulp_video_io.data_file_path)
        self.assertEqual('ANY_OUTPUT_PATH/meta_0.gmeta',
                         self.gulp_video_io.meta_file_path)
        self.assertEqual(self.mock_json_serializer,
                         self.gulp_video_io.serializer)

        self.assertEqual(self.gulp_video_io.meta_dict, None)

    def test_get_or_create_dict_not_exists(self):
        self.assertEqual(self.gulp_video_io.get_or_create_dict('ANY_PATH'),
                         defaultdict(lambda: defaultdict(list)))

    def test_get_or_create_dict_exists(self):
        existing_dict_file = os.path.join(self.temp_dir, 'ANY_DICT')
        open(existing_dict_file, 'w').close()
        self.gulp_video_io.get_or_create_dict(existing_dict_file)
        self.mock_json_serializer.load.called_once_with(existing_dict_file)

    def test_open_with_wb(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            with self.gulp_video_io.open('wb'):
                m.assert_called_once_with(
                    self.gulp_video_io.data_file_path, 'wb')
                get_mock.assert_has_calls([mock.call(
                    self.gulp_video_io.meta_file_path)])

    def test_open_with_rb(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            with self.gulp_video_io.open('rb'):
                m.assert_called_once_with(
                    self.gulp_video_io.data_file_path, 'rb')
                get_mock.assert_has_calls([mock.call(
                    self.gulp_video_io.meta_file_path)])

    def test_open_unknown_flag(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        with self.assertRaises(NotImplementedError):
            with self.gulp_video_io.open('NO_SUCH_FLAG'):
                pass

    def test_flush(self):
        meta_path = os.path.join(self.temp_dir, 'meta_0.gmeta')
        self.gulp_video_io.meta_file_path = meta_path
        self.gulp_video_io.serializer = json_serializer
        self.gulp_video_io.meta_dict = {'0': {'meta_data': []}}
        self.gulp_video_io.flush()
        meta_path_written = open(meta_path).read()
        self.assertEqual('{"0": {"meta_data": []}}', meta_path_written)

#     def test_close_when_open(self):
#         f_mock = mock.Mock()
#         flush_mock = mock.Mock()
#         self.gulp_video_io.f = f_mock
#         self.gulp_video_io.flush = flush_mock
#         self.gulp_video_io.is_open = True
#         self.gulp_video_io.close()
#         self.assertEqual(self.gulp_video_io.is_open, False)
#         f_mock.close.assert_called_once_with()
#         flush_mock.assert_called_once_with()
#
#     def test_close_when_closed(self):
#         self.gulp_video_io.is_open = False
#         self.gulp_video_io.close()

    def test_append_meta(self):
        self.gulp_video_io.meta_dict = {'0': {'meta_data': []}}
        self.gulp_video_io.append_meta(0, {'meta': 'ANY_META'})
        expected = {'0': {'meta_data': [{'meta': 'ANY_META'}]}}
        self.assertEqual(expected, self.gulp_video_io.meta_dict)

    def test_write_frame(self):
        bio = BytesIO()
        self.gulp_video_io.meta_dict = {'0': {'meta_data': [],
                                              'frame_info': []}}
        fp = bio
        with mock.patch('cv2.imencode') as imencode_mock:
            imencode_mock.return_value = '', numpy.ones((1,), dtype='uint8')
            self.gulp_video_io.write_frame(fp, 0, None)
            self.assertEqual(b'\x01\x00\x00\x00', bio.getvalue())
            expected = {'0': {'meta_data': [],
                              'frame_info': [ImgInfo(0, 3, 4)]}}
            self.assertEqual(expected, self.gulp_video_io.meta_dict)

    def test_initialize_filenames(self):
        expected = (self.gulp_video_io.output_path + '/data_0.gulp',
                    self.gulp_video_io.output_path + '/meta_0.gmeta')
        outcome = self.gulp_video_io.initialize_filenames(0)
        self.assertEqual(expected, outcome)

#     def test_read_frame(self):
#         # use 'write_frame' to write a single image
#         bio = BytesIO()
#         self.gulp_video_io.meta_dict = {'0': {'meta_data': [],
#                                               'frame_info': []}}
#         fp = bio
#         image = numpy.ones((3, 3, 3), dtype='uint8')
#         self.gulp_video_io.write_frame(fp, 0, image)
#
#         # recover the single frame using 'read'
#         info = self.gulp_video_io.meta_dict['0']['frame_info'][0]
#         result = self.gulp_video_io.read_frame(fp, info)
#         npt.assert_array_equal(image, numpy.array(result))


class ChunkWriterElement(unittest.TestCase):

    def setUp(self):
        self.adapter = mock.MagicMock()
        self.adapter.__len__.return_value = 1
        self.output_folder = 'ANY_OUTPUT_FOLDER'
        self.videos_per_chunk = 1
        self.chunk_writer = ChunkWriter(self.adapter,
                                        self.output_folder,
                                        self.videos_per_chunk)

    def tearDown(self):
        pass


class TestChunkWriter(ChunkWriterElement):

    def test_initialization(self):
        self.assertEqual(self.adapter,
                         self.chunk_writer.adapter)
        self.assertEqual(self.output_folder,
                         self.chunk_writer.output_folder)
        self.assertEqual(self.videos_per_chunk,
                         self.chunk_writer.videos_per_chunk)

    @mock.patch('gulpio.fileio.GulpChunk')
    def test_write_chunk(self, mock_gulp):
        def mock_iter_data(input_slice):
            yield {'id': 0,
                   'meta': {'meta': 'ANY_META'},
                   'frames': ['ANY_FRAME1', 'ANY_FRAME2'],
                   }
        self.adapter.iter_data = mock_iter_data
        mock_gulp.open = mock.Mock()
        self.chunk_writer.write_chunk((0, 1), 0)
        mock_gulp().write_frame.assert_has_calls(
            [mock.call(mock.call, 0, 'ANY_FRAME1'),
             mock.call(mock.call(), 0, 'ANY_FRAME2')]
        )


class GulpIngestorElement(unittest.TestCase):

    @mock.patch('gulpio.adapters.AbstractDatasetAdapter')
    def setUp(self, mock_adapter):
        self.adapter = mock_adapter
        self.output_folder = 'ANY_OUTPUT_FOLDER'
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

    @mock.patch('gulpio.utils.ensure_output_dir_exists')
    @mock.patch('gulpio.fileio.ChunkWriter')
    @mock.patch('gulpio.fileio.ProcessPoolExecutor')
    def test_ingest(self,
                    mock_process_pool,
                    mock_chunk_writer,
                    mock_ensure_output_dir):

        # The next three lines mock the ProcessPoolExecutor and it's map
        # function.
        executor_mock = mock.Mock()
        executor_mock.map.return_value = []
        mock_process_pool.return_value.__enter__.return_value = executor_mock

        mock_chunk_writer.return_value.__len__.return_value = 2
        mock_chunk_writer.return_value.chunks = [(0, 1), (1, 2)]

        self.gulp_ingestor()
        mock_chunk_writer.assert_called_once_with(self.adapter,
                                                  self.output_folder,
                                                  self.videos_per_chunk,
                                                  )
        executor_mock.map.assert_called_once_with(
            mock_chunk_writer.return_value.write_chunk,
            [(0, 1), (1, 2)],
            range(2),
        )


class RoundTripAdapter(AbstractDatasetAdapter):

    def __init__(self):
        self.result1 = {
            'meta': {'name': 'empty video'},
            'frames': [],
            'id': 0,
        }
        self.result2 = {
            'meta': {'name': 'bunch of numpy arrays'},
            'frames': [
                numpy.ones((4, 1, 3), dtype='uint8'),
                numpy.ones((3, 1, 3), dtype='uint8'),
                numpy.ones((2, 1, 3), dtype='uint8'),
                numpy.ones((1, 1, 3), dtype='uint8'),
            ],
            'id': 1,
        }

    def __len__(self):
        return 2

    def iter_data(self, slice_element=None):
        yield self.result1
        yield self.result2


class TestRoundTrip(FSBase):

    def test(self):
        adapter = RoundTripAdapter()
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()
        gc = GulpChunk(0, output_directory, 1)
        expected_output_shapes = [[(4, 1, 3),
                                   (3, 1, 3),
                                   (2, 1, 3),
                                   (1, 1, 3)]
                                  ]
        expected_meta = [{'name': 'bunch of numpy arrays'}]
        with gc.open('rb') as ch_p:
            for i, id_ in enumerate(sorted(gc.meta_dict.keys())):
                frames, meta = gc.read_frames(ch_p, id_)
                output_shapes = [numpy.array(frame).shape for frame in frames]
                self.assertEqual(expected_meta[i], meta)
                self.assertEqual(expected_output_shapes[i], output_shapes)

        with gc.open('rb') as ch_p:
            chunk_element = gc.read_chunk(ch_p)
            for i, (frames_, meta_) in enumerate(chunk_element):
                self.assertEqual(expected_meta[i], meta_)
                self.assertEqual(expected_output_shapes[i],
                                 [numpy.array(frame).shape
                                  for frame in frames_])
