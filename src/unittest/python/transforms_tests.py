import numpy as np
import cv2
import unittest
from gulpio.transforms import Scale
from gulpio.transforms import RandHorFlipVideo
from gulpio.transforms import Normalize
from gulpio.transforms import UnitNorm
from gulpio.transforms import CenterCrop
from gulpio.transforms import RandomCropVideo
from gulpio.transforms import JitterCropVideo
from gulpio.transforms import ComposeVideo



class TestTransforms(unittest.TestCase):


    def _img(self):
        return np.random.randint(0, 255, [120, 60, 3]).astype('uint8')


    def _img_gray(self):
        return np.random.randint(0, 255, [120, 60]).astype('uint8')


    def _video(self, length):
        imgs = []
        for i in range(length):
            img = np.random.randint(0, 255, [60, 120, 3]).astype('uint8')
            imgs.append(img)
        return imgs


    def _check_video_size(self, video):
        ref_size = video[0].shape
        for img in video:
            assert ref_size[0] == img.shape[0] and ref_size[1] == img.shape[1]


    def test_scale(self):
        img = self._img()
        scale = Scale(30)
        img_out = scale(img)
        assert img_out.shape[0] == 60 and img_out.shape[1] == 30
        assert img_out.shape[2] == 3

        img = self._img()
        scale = Scale((30,30))
        img_out = scale(img)
        assert img_out.shape[0] == 30 and img_out.shape[1] == 30
        assert img_out.shape[2] == 3

        img = self._img_gray()
        scale = Scale(30)
        img_out = scale(img)
        assert img_out.shape[0] == 60 and img_out.shape[1] == 30
        assert img_out.ndim == 2

        video = self._video(10)
        for img in video:
            scale = Scale(30)
            img_out = scale(img)
            assert img_out.shape[0] == 30 and img_out.shape[1] == 60
            assert img_out.shape[2] == 3


    def test_randhorflipvideo(self):
        video = self._video(10)
        flip = RandHorFlipVideo()
        video = flip(video)
        assert video[0].ndim == 3
        assert len(video) == 10


    def test_normalize(self):
        img = np.ones([120, 120, 3])
        norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = norm(img)
        print(img.mean())
        assert img.mean() > 1 - 1e-3 and img.mean() < 1 + 1e-3


    def test_unitnorm(self):
        img = np.ones([120, 120, 3])
        norm = UnitNorm()
        img = norm(img)
        print(img.mean())
        assert img.mean() > -1e-3 and img.mean() < 1e-3


    def test_randomcrop(self):
        video = self._video(10)
        crop = RandomCropVideo(30)
        video = crop(video)
        assert video[0].shape[0] == 30 and video[1].shape[1] == 30
        assert video[0].ndim == 3
        self._check_video_size(video)


    def test_jittercrop(self):
        video = self._video(10)
        crop = JitterCropVideo([30, 20])
        video = crop(video)
        print(video[0].shape)
        assert (video[0].shape[0] + video[1].shape[1] == 60) or \
        (video[0].shape[0] + video[1].shape[1] == 50) or \
        (video[0].shape[0] + video[1].shape[1] == 40)
        assert video[0].ndim == 3
        self._check_video_size(video)


    def test_centercrop(self):
        img = self._img()
        img_gray = self._img_gray()
        crop = CenterCrop(20)
        img = crop(img)
        img_gray = crop(img_gray)
        assert img.shape[0] == 20 and img.shape[1] == 20
        assert img_gray.shape[0] == 20 and img_gray.shape[1] == 20

    def test_composevideo(self):
        img_transforms = [CenterCrop(30)]
        video_transforms = [RandomCropVideo(30)]
        compose = ComposeVideo(img_transforms, video_transforms)
        vid = self._video(10)
        vid = compose(vid)

