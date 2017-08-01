import os
import tempfile
import shutil

from collections import defaultdict
from io import BytesIO

import numpy
import numpy.testing as npt

import unittest
import unittest.mock as mock

from gulpio.fileio import (GulpVideoIO,
                           ChunkWriter,
                           GulpIngestor,
                           calculate_chunks,
                           json_serializer,
                           MetaInfo,
                           ImgInfo,
                           )


class FSBase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='fileio-test-')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


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


class GulpVideoIOElement(FSBase):

    @mock.patch('gulpio.fileio.json_serializer')
    def setUp(self, mock_json_serializer):
        super(GulpVideoIOElement, self).setUp()
        self.mock_json_serializer = mock_json_serializer
        self.path = "ANY_PATH"
        self.meta_path = "ANY_META_PATH"
        self.img_info_path = "ANY_IMG_INFO_PATH"
        self.gulp_video_io = GulpVideoIO(self.path,
                                         self.meta_path,
                                         self.img_info_path,
                                         mock_json_serializer)

    def tearDown(self):
        super(GulpVideoIOElement, self).tearDown()


class TestGulpVideoIO(GulpVideoIOElement):

    def test_initializer(self):
        self.assertEqual(self.path, self.gulp_video_io.path)
        self.assertEqual(self.meta_path, self.gulp_video_io.meta_path)
        self.assertEqual(self.img_info_path, self.gulp_video_io.img_info_path)
        self.assertEqual(self.mock_json_serializer,
                         self.gulp_video_io.serializer)

        self.assertEqual(self.gulp_video_io.is_open, False)
        self.assertEqual(self.gulp_video_io.is_writable, False)
        self.assertEqual(self.gulp_video_io.f, None)
        self.assertEqual(self.gulp_video_io.img_dict, None)
        self.assertEqual(self.gulp_video_io.meta_dict, None)

    def test_get_or_create_dict_not_exists(self):
        self.assertEqual(self.gulp_video_io.get_or_create_dict('ANY_PATH'),
                         defaultdict(list))

    def test_get_or_create_dict_exists(self):
        existing_dict_file = os.path.join(self.temp_dir, 'ANY_DICT')
        open(existing_dict_file, 'w').close()
        self.gulp_video_io.get_or_create_dict(existing_dict_file)
        self.mock_json_serializer.load.called_once_with(existing_dict_file)

    def test_open_with_wb(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            self.gulp_video_io.open('wb')
            m.assert_called_once_with(self.path, 'wb')
            self.assertEqual(self.gulp_video_io.is_writable, True)
            self.assertEqual(self.gulp_video_io.is_open, True)
            get_mock.assert_has_calls([mock.call(self.meta_path),
                                       mock.call(self.img_info_path)])

    def test_open_with_rb(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        with mock.patch('builtins.open', new_callable=mock.mock_open()) as m:
            self.gulp_video_io.open('rb')
            m.assert_called_once_with(self.path, 'rb')
            self.assertEqual(self.gulp_video_io.is_writable, False)
            self.assertEqual(self.gulp_video_io.is_open, True)
            get_mock.assert_has_calls([mock.call(self.meta_path),
                                       mock.call(self.img_info_path)])

    def test_open_unknown_flag(self):
        get_mock = mock.Mock()
        self.gulp_video_io.get_or_create_dict = get_mock
        self.assertRaises(NotImplementedError,
                          self.gulp_video_io.open,
                          'NO_SUCH_FLAG')

    def test_flush(self):
        meta_path = os.path.join(self.temp_dir, self.meta_path)
        self.gulp_video_io.meta_path = meta_path
        img_info_path = os.path.join(self.temp_dir, self.img_info_path)
        self.gulp_video_io.serializer = json_serializer
        self.gulp_video_io.img_info_path = img_info_path
        self.gulp_video_io.meta_dict = {'meta': 'ANY_META'}
        self.gulp_video_io.img_dict = {'img_info': 'ANY_IMG_INFO'}
        self.gulp_video_io.flush()
        meta_path_written = open(meta_path).read()
        self.assertEqual('{"meta": "ANY_META"}', meta_path_written)
        img_info_path_written = open(img_info_path).read()
        self.assertEqual('{"img_info": "ANY_IMG_INFO"}', img_info_path_written)

    def test_close_when_open(self):
        f_mock = mock.Mock()
        flush_mock = mock.Mock()
        self.gulp_video_io.f = f_mock
        self.gulp_video_io.flush = flush_mock
        self.gulp_video_io.is_open = True
        self.gulp_video_io.close()
        self.assertEqual(self.gulp_video_io.is_open, False)
        f_mock.close.assert_called_once_with()
        flush_mock.assert_called_once_with()

    def test_close_when_closed(self):
        self.gulp_video_io.is_open = False
        self.gulp_video_io.close()

    def test_append_meta(self):
        self.gulp_video_io.is_writable = True
        self.gulp_video_io.meta_dict = {0: []}
        self.gulp_video_io.append_meta(0, 0, {'meta': 'ANY_META'})
        expected = {0: [MetaInfo(0, {'meta': 'ANY_META'})]}
        self.assertEqual(expected, self.gulp_video_io.meta_dict)

    def test_write_frame(self):
        self.gulp_video_io.is_writable = True
        bio = BytesIO()
        self.gulp_video_io.img_dict = {0: []}
        self.gulp_video_io.f = bio
        with mock.patch('cv2.imencode') as imencode_mock:
            imencode_mock.return_value = '', numpy.ones((1,), dtype='uint8')
            self.gulp_video_io.write_frame(0, 0, None)
            self.assertEqual(b'\x01\x00\x00\x00', bio.getvalue())
            expected = {0: [ImgInfo(0, 3, 4)]}
            self.assertEqual(expected, self.gulp_video_io.img_dict)

    def test_read_frame(self):
        # use 'write_frame' to write a single image
        self.gulp_video_io.is_writable = True
        bio = BytesIO()
        self.gulp_video_io.img_dict = {0: []}
        self.gulp_video_io.f = bio
        image = numpy.ones((3, 3, 3), dtype='uint8')
        self.gulp_video_io.write_frame(0, 0, image)

        # recover the single frame using 'read'
        self.gulp_video_io.is_writable = False
        info = self.gulp_video_io.img_dict[0][0]
        result = self.gulp_video_io.read_frame(info)
        npt.assert_array_equal(image, numpy.array(result))


class ChunkWriterElement(unittest.TestCase):

    @mock.patch('gulpio.adapters.AbstractDatasetAdapter')
    def setUp(self, mock_adapter):
        self.adapter = mock_adapter
        self.output_folder = 'ANY_OUTPUT_FOLDER'
        self.chunk_writer = ChunkWriter(self.adapter, self.output_folder)

    def tearDown(self):
        pass


class TestChunkWriter(ChunkWriterElement):

    def test_initialization(self):
        self.assertEqual(self.adapter, self.chunk_writer.adapter)
        self.assertEqual(self.output_folder, self.chunk_writer.output_folder)

    def test_initialize_filenames(self):
        expected = (self.output_folder + '/data0.bin',
                    self.output_folder + '/img_info0.bin',
                    self.output_folder + '/meta0.bin')
        outcome = self.chunk_writer.initialize_filenames(0)
        self.assertEqual(expected, outcome)

    @mock.patch('gulpio.fileio.GulpVideoIO')
    def test_write_chunk(self, mock_gulp):
        def mock_iter_data(input_slice):
            yield {'id': 0,
                   'meta': {'meta': 'ANY_META'},
                   'frames': ['ANY_FRAME1', 'ANY_FRAME2'],
                   }
        self.adapter.iter_data = mock_iter_data
        self.chunk_writer.write_chunk((0, 1), 0)
        mock_gulp().write_frame.assert_has_calls(
            [mock.call(0, 0, 'ANY_FRAME1'),
             mock.call(0, 0, 'ANY_FRAME2')]
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
        self.adapter.__len__.return_value = 2

        # The next three lines mock the ProcessPoolExecutor and it's map
        # function.
        executor_mock = mock.Mock()
        executor_mock.map.return_value = []
        mock_process_pool.return_value.__enter__.return_value = executor_mock

        self.gulp_ingestor.ingest()
        mock_chunk_writer.assert_called_once_with(self.adapter,
                                                  self.output_folder)
        executor_mock.map.assert_called_once_with(
            mock_chunk_writer().write_chunk,
            [(0, 1), (1, 2)],
            range(2),
            chunksize=1,
        )

