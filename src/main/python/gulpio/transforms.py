from __future__ import division
import random
import numbers
import collections
import numpy as np
import cv2


class Compose(object):
    r"""Composes several transforms together. It is for image problems

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms = [transforms.Normalize()]
        >>> transforms.Compose(transforms)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ComposeVideo(object):
    r"""Composes several transforms together. It takes two lists of
    transformations. One list is for image transformations, and the other one
    is for videos. Image transforms are called per frame and video
    transforms applied to the whole video with the same parameters.

    Args:
        img_transforms (List[Transform]): list of transforms to compose.
        video_transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> img_transforms = [transforms.Normalize()]
        >>> video_transforms = [transforms.CenterCrop(224)]
        >>> transforms.ComposeVideo(img_transforms, video_transforms)
    """

    def __init__(self, img_transforms=[], video_transforms=[]):
        self.img_transforms = img_transforms
        self.video_transforms = video_transforms

    def __call__(self, imgs):
        for t in self.img_transforms:
            for idx, img in enumerate(imgs):
                imgs[idx] = t(img)
        for t in self.video_transforms:
            imgs = t(imgs)
        return imgs


class RandHorFlipVideo():
    r""" Apply random horizontal flip to video """

    def __call__(self, imgs):
        if random.random() < 0.5:
            for idx, img in enumerate(imgs):
                img = cv2.flip(img, 1)
                imgs[idx] = img
        return imgs


class RandVerFlipVideo():
    r""" Apply random vertical flip to video """

    def __call__(self, imgs):
        if random.random() < 0.5:
            for idx, img in enumerate(imgs):
                img = cv2.flip(img, 0)
                imgs[idx] = img
        return imgs


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # Relying on Numpy broadcasting abilities
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class UnitNorm(object):
    r"""Instance wise unit norm"""

    def __init__(self):
        return

    def __call__(self, tensor):
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor


class CenterCrop(object):
    r"""Crops the given image at the center. It uses OpenCV.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = Scale(size)

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be cropped.
        Returns:
            numpy.array: Cropped image.
        """
        h, w = img.shape[0], img.shape[1]
        if self.size[0] > w or self.size[1] > h:
            img = self.scale(img)
            w, h = img.size

        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1: y1 + th, x1: x1 + tw]


class RandomCrop(object):
    """Crop the given image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be cropped.
        Returns:
            numpy.array: Cropped image.
        """
        th, tw = self.size
        h, w = img.shape[:2]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if self.padding > 0:
            img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                     self.padding, self.padding,
                                     cv2.BORDER_CONSTANT, value=0)
        # sample crop locations if not given
        # it is necessary to keep cropping same in a video
        img_crop = img[y1:y1+th, x1:x1+tw]
        return img_crop


class RandomCropVideo(object):
    """Crop the given video frames at a random location. Crop location is the
    same for all the frames.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Image to be cropped.
        Returns:
            numpy.array: Cropped image.
        """
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         cv2.BORDER_CONSTANT, value=0)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1+th, x1:x1+tw]
            imgs[idx] = img_crop
        return imgs


class JitterCrop(object):
    """Random cropping with pre-defined set of w and h.
    Args:
        sample_sizes (sequence, optional): possible crop sizes.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, sample_sizes=[256, 224, 192, 168], padding=0):
        self.padding = padding
        self.sample_sizes = sample_sizes

    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Image to be cropped.
        Returns:
            numpy.array: Cropped image.
        """
        sample_w = random.choice(self.sample_sizes)
        sample_h = random.choice(self.sample_sizes)
        h, w = imgs[0].shape[:2]
        x1 = random.randint(0, w - sample_w)
        y1 = random.randint(0, h - sample_h)
        if self.padding > 0:
            img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                     self.padding, self.padding,
                                     cv2.BORDER_CONSTANT, value=0)
        # sample crop locations if not given
        # it is necessary to keep cropping same in a video
        img_crop = img[y1:y1+sample_h, x1:x1+sample_w]
        return img_crop


class JitterCropVideo(object):
    """Random cropping with pre-defined set of w and h.
    Args:
        sample_sizes (sequence, optional): possible crop sizes.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, sample_sizes=[256, 224, 192, 168], padding=0):
        self.padding = padding
        self.sample_sizes = sample_sizes

    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Image to be cropped.
        Returns:
            numpy.array: Cropped image.
        """
        sample_w = random.choice(self.sample_sizes)
        sample_h = random.choice(self.sample_sizes)
        h, w = imgs[0].shape[:2]
        x1 = random.randint(0, w - sample_w)
        y1 = random.randint(0, h - sample_h)
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         cv2.BORDER_CONSTANT, value=0)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1+sample_h, x1:x1+sample_w]
            imgs[idx] = img_crop
        return imgs


class Scale(object):
    """Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be scaled.
        Returns:
            numpy.array: Rescaled image.
        """
        if isinstance(self.size, int):
            h, w = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                if ow < w:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if oh < h:
                    return cv2.resize(img, (ow, oh), cv2.INTER_AREA)
                else:
                    return cv2.resize(img, (ow, oh))
        else:
            return cv2.resize(img, tuple(self.size))
