import os
import glob
import cv2
import numpy as np
import json


def find_gulp_files(folder):
    chunk_paths = sorted(glob.glob(os.path.join(folder, 'data*.gulp')))
    meta_paths = sorted(glob.glob(os.path.join(folder, 'meta*.gmeta')))
    label2idx = json.load(open(os.path.join(folder, 'label2idx.json'), 'rb'))
    assert len(chunk_paths) == len(meta_paths)
    return (chunk_paths, meta_paths, label2idx)


def merge_meta_files(meta_paths):
    meta = {}
    for meta_path in meta_paths:
        meta_temp = json.load(open(meta_path, 'rb'))
        for key in meta_temp.keys():
            chunk = os.path.basename(meta_path).replace('gmeta', 'gulp')\
                    .replace('meta', 'data')
            meta_temp[key]['chunk_file'] = chunk
        meta.update(meta_temp)
    return list(meta.items())


class GulpIOEmptyFolder(Exception):  # pragma: no cover
        pass


class GulpIOMismatch(Exception):
        pass


class GulpVideoDataset(object):

    def __init__(self, data_path, num_frames, step_size,
                 is_val, transform=None, target_transform=None, stack=True):
        r"""Simple data loader for GulpIO format.

            Args:
                data_path (str): path to GulpIO dataset folder
                label_path (str): path to GulpIO label dictionary matching
            label ids to label names
                num_frames (int): number of frames to be fetched.
                step_size (int): number of frames skippid while picking
            sequence of frames from each video.
                is_va (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                target_transform (func): performs preprocessing on labels if
            defined. Default is None.
                stack (bool): stack frames into a numpy.array. Default is True.
        """

        self.chunk_paths, self.meta_paths, self.label_to_idx = find_gulp_files(data_path)
        self.num_chunks = len(self.chunk_paths)

        if len(self.chunk_paths) == 0:
            raise(GulpIOEmptyFolder(r"Found 0 data binaries in subfolders \
                                    of: ".format(data_path)))

        if len(self.chunk_paths) != len(self.meta_paths):
            raise(GulpIOMismatch(r"Number of binary files are not matching \
                                 with number of meta files. Check GulpIO \
                                 dataset."))

        print(" > Found {} chunks".format(self.num_chunks))
        self.data_path = data_path
        self.meta_dict = merge_meta_files(self.meta_paths)
        self.classes = self.label_to_idx.keys()
        self.transform_video = transform
        self.target_transform = target_transform
        self.num_frames = num_frames
        self.step_size = step_size
        self.is_val = is_val
        self.stack = stack

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        item_idx, item_info = self.meta_dict[index]
        chunk_path = os.path.join(self.data_path, item_info['chunk_file'])
        chunk_file = open(chunk_path, "rb")
        target_name = item_info['meta_data'][0]['label']
        target_idx = self.label_to_idx[target_name]
        frames = item_info['frame_info']
        num_frames = len(frames)
        # set number of necessary frames
        if self.num_frames > -1:
            num_frames_necessary = self.num_frames * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frames.extend([frames[-1]] * (num_frames_necessary - num_frames))
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        # set target frames to be loaded
        frames = frames[offset: num_frames_necessary + offset: self.step_size]
        # read images
        imgs = []
        for frame in frames:
            img = self.__read_frame(frame, chunk_file)
            imgs.append(img)
        # augmentation
        if self.transform_video:
            imgs = self.transform_video(imgs)
        # format data to torch tensor
        chunk_file.close()
        if self.stack:
            imgs = np.stack(imgs)
        return (imgs, target_idx)

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.meta_dict)

    def __read_frame(self, meta_info, f):
        """
        Reads single frames from a video
        """
        # TODO: read item by gulpio api
        loc, pad, length = meta_info
        f.seek(loc)
        record = f.read(length)
        img_str = record[:-pad]
        nparr = np.fromstring(img_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
