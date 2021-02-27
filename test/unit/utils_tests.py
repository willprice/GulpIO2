import os
import tempfile
import shutil

import numpy as np
import numpy.testing as npt

import unittest
import unittest.mock as mock

from gulpio2.utils import (
    check_ffmpeg_exists,
    burst_video_into_frames,
    img_to_jpeg_bytes,
    jpeg_bytes_to_img,
    resize_by_short_edge,
    resize_images,
    get_single_video_path,
    temp_dir_for_bursting,
    DuplicateIdException,
    remove_entries_with_duplicate_ids,
    _remove_duplicates_in_metadict,
)
from gulpio2.fileio import GulpIngestor
from fileio_tests import DummyVideosAdapter
from fileio_tests import create_image


class FSBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="utils-test-")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestReadingAndWritingJpegs(unittest.TestCase):
    def test_rgb_image_round_trip(self):
        img = np.zeros((15, 15, 3), dtype=np.uint8)
        img[..., 1] = 127
        img[..., 2] = 255
        npt.assert_allclose(
                jpeg_bytes_to_img(img_to_jpeg_bytes(img)), img, atol=3
        )

    def test_gray_image_round_trip(self):
        img = np.zeros((15, 15), dtype=np.uint8)
        img[5:5] = 128
        img[10:15] = 256
        npt.assert_allclose(
                jpeg_bytes_to_img(img_to_jpeg_bytes(img)), img, atol=3
        )


class TestCheckFFMPEGExists(unittest.TestCase):
    @mock.patch("os.system", mock.Mock(return_value=0))
    def test_exists(self):
        self.assertEqual(True, check_ffmpeg_exists())

    @mock.patch("os.system", mock.Mock(return_value=1))
    def test_does_not_exists(self):
        self.assertEqual(False, check_ffmpeg_exists())


class TestBurstVideoIntoFrames(unittest.TestCase):
    def test_mp4(self):
        video_path = os.path.join(os.path.dirname(__file__), "test.mp4")
        with temp_dir_for_bursting() as temp_burst_dir:
            imgs = burst_video_into_frames(video_path, temp_burst_dir)
        # different ffmpeg versions, yield slightly different numbers
        self.assertIn(len(imgs), [140, 141])

    def test_mp4_with_lower_frame_rate(self):
        video_path = os.path.join(os.path.dirname(__file__), "test.mp4")
        with temp_dir_for_bursting() as temp_burst_dir:
            imgs = burst_video_into_frames(video_path, temp_burst_dir, frame_rate=8)
        self.assertEqual(39, len(imgs))

    def test_webm(self):
        video_path = os.path.join(os.path.dirname(__file__), "test.webm")
        with temp_dir_for_bursting() as temp_burst_dir:
            imgs = burst_video_into_frames(video_path, temp_burst_dir)
        # different ffmpeg versions, yield slightly different numbers
        self.assertIn(len(imgs), [140, 141])

    def test_webm_with_lower_frame_rate(self):
        video_path = os.path.join(os.path.dirname(__file__), "test.webm")
        with temp_dir_for_bursting() as temp_burst_dir:
            imgs = burst_video_into_frames(video_path, temp_burst_dir, frame_rate=8)
        self.assertEqual(39, len(imgs))


class TestResizeImages(unittest.TestCase):
    @mock.patch("PIL.Image.open")
    @mock.patch("gulpio2.utils.resize_by_short_edge")
    def test(self, mock_resize, mock_image_open):
        mock_image_open.side_effect = ["READ_IMAGE1", "READ_IMAGE2", "READ_IMAGE3"]
        mock_resize.side_effect = ["RESIZED_IMAGE1", "RESIZED_IMAGE2", "RESIZED_IMAGE3"]
        input_ = ["ANY_IMAGE1", "ANY_IMAGE2", "ANY_IMAGE3"]
        received = list(resize_images(input_, img_size=1))
        self.assertEqual(
            ["RESIZED_IMAGE1", "RESIZED_IMAGE2", "RESIZED_IMAGE3"], received
        )


class TestResizeByShortEdge(unittest.TestCase):
    def test_resize_first_edge_shorter(self):
        input_image = create_image((6, 10), val=0)
        size = 3
        correct_result = create_image((3, 5), val=0)
        result = resize_by_short_edge(input_image, size)
        np.testing.assert_array_equal(correct_result, result)

    def test_resize_second_edge_shorter(self):
        input_image = create_image((10, 6))
        size = 3
        correct_result = create_image((5, 3))
        result = resize_by_short_edge(input_image, size)
        np.testing.assert_array_equal(correct_result, result)


class TestGetSingleVideoPath(FSBase):
    def test_video_exists(self):
        test_video_path = os.path.join(self.temp_dir, "test.mp4")
        open(test_video_path, "w").close()
        received = get_single_video_path(self.temp_dir)
        self.assertEqual(test_video_path, received)

    def test_video_doesnt_exists(self):
        self.assertRaises(AssertionError, get_single_video_path, "ANY_PATH")


class TestRemoveEntriesWithDuplicateIds(FSBase):
    def test_no_duplicates(self):
        adapter = DummyVideosAdapter(3)
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()
        meta_dict = [
            {
                "meta": {"name": "new_video"},
                "frames": [np.ones((4, 1, 3), dtype="uint8")],
                "id": 3,
            }
        ]
        new_meta = remove_entries_with_duplicate_ids(output_directory, meta_dict)
        self.assertEqual(meta_dict, new_meta)

    def test_one_out_of_one_duplicate(self):
        adapter = DummyVideosAdapter(3)
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()
        meta_dict = [
            {
                "meta": {"name": "new_video"},
                "frames": [np.ones((4, 1, 3), dtype="uint8")],
                "id": 1,
            }
        ]
        with self.assertRaises(DuplicateIdException):
            remove_entries_with_duplicate_ids(output_directory, meta_dict)

    def test_one_out_of_two_duplicate(self):
        adapter = DummyVideosAdapter(3)
        output_directory = os.path.join(self.temp_dir, "ANY_OUTPUT_DIR")
        ingestor = GulpIngestor(adapter, output_directory, 2, 1)
        ingestor()
        input1 = {
            "meta": {"name": "new_video"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 1,
        }
        input2 = {
            "meta": {"name": "new_videoi_2"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 3,
        }
        meta_dict = [input1, input2]
        new_meta = remove_entries_with_duplicate_ids(output_directory, meta_dict)
        self.assertEqual(new_meta, [input2])


class TestRemoveDuplicatesInMetadict(unittest.TestCase):
    def test_no_duplicates_in_metadict(self):
        input1 = {
            "meta": {"name": "new_video"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 1,
        }
        input2 = {
            "meta": {"name": "new_video2"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 2,
        }
        meta = [input1, input2]
        new_meta = _remove_duplicates_in_metadict(meta)
        self.assertEqual(meta, new_meta)

    def test_duplicates_in_metadict(self):
        input1 = {
            "meta": {"name": "new_video"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 1,
        }
        input2 = {
            "meta": {"name": "new_video2"},
            "frames": [np.ones((4, 1, 3), dtype="uint8")],
            "id": 2,
        }
        meta = [input1, input1, input2]
        new_meta = _remove_duplicates_in_metadict(meta)
        self.assertEqual([input1, input2], new_meta)
