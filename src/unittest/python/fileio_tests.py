import os
import tempfile
import shutil
import json

import unittest
import unittest.mock as mock

from gulpio.fileio import (calculate_chunks,
                           GulpVideoIO,
                           json_serializer
                           )


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


class TestGulpVideoIO(unittest.TestCase):
    @mock.patch('gulpio.fileio.json_serializer')
    def test_initializer(self, mock_json_serializer):
        path = "ANY_PATH"
        flag = "ANY_FLAG"
        meta_path = "ANY_META_PATH"
        img_info_path = "ANY_IMG_INFO_PATH"
        gulpio_video_io = GulpVideoIO(path, flag, meta_path, img_info_path,
                                      mock_json_serializer)
        self.assertEqual(path, gulpio_video_io.path)
        self.assertEqual(flag, gulpio_video_io.flag)
        self.assertEqual(meta_path, gulpio_video_io.meta_path)
        self.assertEqual(img_info_path, gulpio_video_io.img_info_path)
        self.assertEqual(mock_json_serializer, gulpio_video_io.serializer)

        self.assertEqual(gulpio_video_io.is_open, False)
        self.assertEqual(gulpio_video_io.is_writable, False)
        self.assertEqual(gulpio_video_io.f, None)
        self.assertEqual(gulpio_video_io.img_dict, None)
        self.assertEqual(gulpio_video_io.meta_dict, None)
