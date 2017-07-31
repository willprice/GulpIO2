import os
import tempfile
import shutil

from collections import defaultdict

import unittest
import unittest.mock as mock

from gulpio.fileio import (calculate_chunks,
                           GulpVideoIO,
                           )


class FSBase(unittest.TestCase):

    def setUp(self):
        print("setup ##############################################")
        self.temp_dir = tempfile.mkdtemp(prefix='fileio_test-')

    def tearDown(self):
        print("tearDown ##########################################")
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

    def test_no_videos_in_chunk(self):
        pass

    def test_num_videos_is_zero(self):
        pass


class GulpVideoIOElement(FSBase):

    @mock.patch('gulpio.fileio.json_serializer')
    def setUp(self, mock_json_serializer):
        super(GulpVideoIOElement, self).setUp()
        self.mock_json_serializer = mock_json_serializer
        self.path = "ANY_PATH"
        self.flag = "ANY_FLAG"
        self.meta_path = "ANY_META_PATH"
        self.img_info_path = "ANY_IMG_INFO_PATH"
        self.gulp_video_io = GulpVideoIO(self.path,
                                         self.flag,
                                         self.meta_path,
                                         self.img_info_path,
                                         mock_json_serializer)

    def tearDown(self):
        pass


class TestGulpVideoIO(GulpVideoIOElement):

    def test_initializer(self):
        self.assertEqual(self.path, self.gulp_video_io.path)
        self.assertEqual(self.flag, self.gulp_video_io.flag)
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






