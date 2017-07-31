import os
import tempfile
import shutil
import json

import unittest
import unittest.mock as mock

from gulpio.fileio import (calculate_chunks,
                           GulpVideoIO,
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


