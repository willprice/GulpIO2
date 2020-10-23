import os
import sys
import tempfile
import shutil
from collections import OrderedDict
from contextlib import contextmanager
import io

import unittest
import unittest.mock as mock

from gulpio2.sanity_utils import (check_meta_file_size_larger_zero,
                                 check_data_file_size_larger_zero,
                                 check_data_file_size,
                                 check_for_duplicate_ids,
                                 extract_all_ids,
                                 get_duplicate_entries,
                                 check_for_failures,
                                 )


class FSBase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='sanity-utils-test-')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


@contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestCheckMetaFileSizeLargerZero(FSBase):

    def test_check_meta_file_size_one_empty(self):
        meta_file_path_1 = os.path.join(self.temp_dir, "EMPTY")
        meta_file_path_2 = os.path.join(self.temp_dir, "NOT_EMPTY")
        open(meta_file_path_1, 'w').close()
        with open(meta_file_path_2, 'w') as f:
            f.write("ANY")
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        chunk1.meta_file_path = meta_file_path_1
        chunk2.meta_file_path = meta_file_path_2
        result = check_meta_file_size_larger_zero(gulp_directory)
        self.assertEqual([meta_file_path_1], result)

    def test_check_meta_file_size_no_empty(self):
        meta_file_path_1 = os.path.join(self.temp_dir, "NOT_EMPTY1")
        meta_file_path_2 = os.path.join(self.temp_dir, "NOT_EMPTY2")
        with open(meta_file_path_1, 'w') as f:
            f.write("ANY")
        with open(meta_file_path_2, 'w') as f:
            f.write("ANY")
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        chunk1.meta_file_path = meta_file_path_1
        chunk2.meta_file_path = meta_file_path_2
        result = check_meta_file_size_larger_zero(gulp_directory)
        self.assertEqual([], result)


class TestCheckDataaFileSizeLargerZero(FSBase):

    def test_check_data_file_size_one_empty(self):
        data_file_path_1 = os.path.join(self.temp_dir, "EMPTY")
        data_file_path_2 = os.path.join(self.temp_dir, "NOT_EMPTY")
        open(data_file_path_1, 'w').close()
        with open(data_file_path_2, 'w') as f:
            f.write("ANY")
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        chunk1.data_file_path = data_file_path_1
        chunk2.data_file_path = data_file_path_2
        result = check_data_file_size_larger_zero(gulp_directory)
        self.assertEqual([data_file_path_1], result)

    def test_check_data_file_size_no_empty(self):
        data_file_path_1 = os.path.join(self.temp_dir, "NOT_EMPTY1")
        data_file_path_2 = os.path.join(self.temp_dir, "NOT_EMPTY2")
        with open(data_file_path_1, 'w') as f:
            f.write("ANY")
        with open(data_file_path_2, 'w') as f:
            f.write("ANY")
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        chunk1.data_file_path = data_file_path_1
        chunk2.data_file_path = data_file_path_2
        result = check_data_file_size_larger_zero(gulp_directory)
        self.assertEqual([], result)


class TestCheckDataFileSize(FSBase):

    def test_correct_data_file_size(self):
        gulp_directory = mock.Mock()
        chunk = mock.Mock()
        gulp_directory.chunks.return_value = [chunk]
        chunk.meta_dict = OrderedDict(
            [("0", {"meta_data": [{"ANY0": "META0"}],
                    "frame_info": [[0, 1, 2], [2, 1, 2]]}),
             ("1", {"meta_data": [{"ANY1": "META1"}],
                    "frame_info": [[4, 2, 2], [8, 1, 2]]})])
        data_file_path = os.path.join(self.temp_dir, "10BYTES")
        with open(data_file_path, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09')
        chunk.data_file_path = data_file_path
        result = check_data_file_size(gulp_directory)
        self.assertEqual([], result)

    def test_incorrect_data_file_size(self):
        gulp_directory = mock.Mock()
        chunk = mock.Mock()
        gulp_directory.chunks.return_value = [chunk]
        chunk.meta_dict = OrderedDict(
            [("0", {"meta_data": [{"ANY0": "META0"}],
                    "frame_info": [[0, 1, 2], [2, 1, 2]]})])
        data_file_path = os.path.join(self.temp_dir, "10BYTES")
        with open(data_file_path, 'wb') as f:
            f.write(b'\x00')
        chunk.data_file_path = data_file_path
        result = check_data_file_size(gulp_directory)
        self.assertEqual([data_file_path], result)


class TestCheckForDuplicateIds(unittest.TestCase):

    def test_one_duplicate(self):
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        meta_dict1 = OrderedDict([("0", "ANY"),
                                  ("1", "OTHER")])
        meta_dict2 = OrderedDict([("2", "ANY"),
                                  ("1", "OTHER")])
        chunk1.meta_dict = meta_dict1
        chunk2.meta_dict = meta_dict2

        expected = ["1"]
        result = check_for_duplicate_ids(gulp_directory)
        self.assertEqual(expected, result)

    def test_no_duplicates(self):
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        meta_dict1 = OrderedDict([("0", "ANY"),
                                  ("1", "OTHER")])
        meta_dict2 = OrderedDict([("2", "ANY"),
                                  ("3", "OTHER")])
        chunk1.meta_dict = meta_dict1
        chunk2.meta_dict = meta_dict2

        expected = []
        result = check_for_duplicate_ids(gulp_directory)
        self.assertEqual(expected, result)


class TestExtractAllIds(unittest.TestCase):

    def test_extract_all_ids(self):
        gulp_directory = mock.Mock()
        chunk1 = mock.Mock()
        chunk2 = mock.Mock()
        gulp_directory.chunks.return_value = [chunk1, chunk2]
        meta_dict1 = OrderedDict([("0", "ANY"),
                                  ("1", "OTHER")])
        meta_dict2 = OrderedDict([("2", "ANY"),
                                  ("1", "OTHER")])
        chunk1.meta_dict = meta_dict1
        chunk2.meta_dict = meta_dict2

        expected = ["0", "1", "2", "1"]
        result = extract_all_ids(gulp_directory)
        self.assertEqual(expected, result)


class TestGetDuplicateEntries(unittest.TestCase):

    def test_no_duplicates(self):
        list_ = [0, 1, 2]
        result = get_duplicate_entries(list_)
        self.assertEqual([], result)

    def test_one_duplicates(self):
        list_ = [0, 1, 1]
        result = get_duplicate_entries(list_)
        self.assertEqual([1], result)

    def test_two_duplicates(self):
        list_ = [0, 1, 1, 1]
        result = get_duplicate_entries(list_)
        self.assertEqual([1], result)


class TestCheckForFailures(unittest.TestCase):

    def test_no_failures(self):
        test_result = {"message": "ANY",
                       "failures": []}
        with captured_output() as (out, err):
            check_for_failures(test_result)
        output = out.getvalue().strip()
        expected = "Sanity Check: ANY\nTest passed"
        self.assertEqual(expected, output)

    def test_one_failures(self):
        test_result = {"message": "ANY",
                       "failures": ["FAILURE"]}
        with captured_output() as (out, err):
            check_for_failures(test_result)
        output = out.getvalue().strip()
        expected = "Sanity Check: ANY\nTest failed for: ['FAILURE']"
        self.assertEqual(expected, output)
