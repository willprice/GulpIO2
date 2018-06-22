import os
import json
import unittest
import numpy as np
import tempfile
from gulpio.loader import DataLoader
from gulpio.dataset import GulpVideoDataset, GulpImageDataset
from gulpio.fileio import GulpChunk


class SimpleDataset(object):

    def __getitem__(self, index):
        return np.random.rand(3, 4, 32, 32), np.random.randint(0, high=4, size=4)

    def __len__(self):
        return 12


class TestDataLoader(unittest.TestCase):

    def setup(self):
        self.dataset = SimpleDataset()

    def test_loader(self):
        self.setup()
        loader = DataLoader(self.dataset, num_workers=1)
        for data, label in loader:
            print(data.shape)
        loader = DataLoader(self.dataset, num_workers=2)
        for data, label in loader:
            print(data.shape)
        loader = DataLoader(self.dataset, num_workers=0)
        for data, label in loader:
            print(data.shape)
        loader = DataLoader(self.dataset, num_workers=2, shuffle=True)
        for data, label in loader:
            print(data.shape)
        loader = DataLoader(self.dataset, num_workers=1, drop_last=True)
        for data, label in loader:
            print(data.shape)
        loader = DataLoader(self.dataset, num_workers=1, shuffle=True,
                            drop_last=True)
        for data, label in loader:
            print(data.shape)


class TestGulpVideoDataset(unittest.TestCase):

    def iterate(self, loader):
        for idx, data in enumerate(loader):
            assert len(data) == 2
            assert data[0].sum() == 0
            assert data[1][0] == 1
            if idx == 5:
                break

    def create_chunk(self):
        self.temp_dir = tempfile.mkdtemp(prefix='gulpio-loader-test-')
        self.chunk_path = os.path.join(self.temp_dir, 'data_0.gulp')
        self.meta_path = os.path.join(self.temp_dir, 'meta_0.gmeta')
        self.json_path = os.path.join(self.temp_dir, 'label2idx.json')
        label2dict = {"0": 1, "1": 2, "2": 2}
        json.dump(label2dict, open(self.json_path, 'w'))
        meta_info = {"label": "0"}
        frame = np.zeros([100, 100, 3])  # create a zero pixels image
        chunk = GulpChunk(self.chunk_path, self.meta_path)
        with chunk.open('wb'):
            for i in range(128):  # 128 videos
                frames = [frame for j in range(32)]  # 32 frames
                chunk.append(i, meta_info, frames)

    def test_dataset(self):
        self.create_chunk()
        dataset = GulpVideoDataset(self.temp_dir, 2, 2,
                                   False)
        loader = DataLoader(dataset, num_workers=0)
        self.iterate(loader)

        loader = DataLoader(dataset, num_workers=2)
        self.iterate(loader)

        dataset = GulpVideoDataset(self.temp_dir, 2, 2,
                                   False, stack=False)
        self.iterate(loader)

    def test_target_transform(self):
        self.create_chunk()
        target_label = 7
        dataset = GulpVideoDataset(self.temp_dir, 2, 2,
                                   False, target_transform=lambda y: target_label)
        assert dataset[0][1] == target_label


class TestGulpImageDataset(unittest.TestCase):

    def iterate(self, loader):
        for idx, data in enumerate(loader):
            assert len(data) == 2
            assert data[0].sum() == 0
            assert data[1][0] == 1
            if idx == 5:
                break

    def create_chunk(self):
        self.temp_dir = tempfile.mkdtemp(prefix='gulpio-loader-test-')
        self.chunk_path = os.path.join(self.temp_dir, 'data_0.gulp')
        self.meta_path = os.path.join(self.temp_dir, 'meta_0.gmeta')
        self.json_path = os.path.join(self.temp_dir, 'label2idx.json')
        label2dict = {"0": 1, "1": 2, "2": 2}
        json.dump(label2dict, open(self.json_path, 'w'))
        meta_info = {"label": "0"}
        frame = np.zeros([100, 100, 3])  # create a zero pixels image
        chunk = GulpChunk(self.chunk_path, self.meta_path)
        with chunk.open('wb'):
            for i in range(128):  # 128 videos
                chunk.append(i, meta_info, [frame])

    def test_dataset(self):
        self.create_chunk()
        dataset = GulpImageDataset(self.temp_dir)
        loader = DataLoader(dataset, num_workers=0)
        self.iterate(loader)

        loader = DataLoader(dataset, num_workers=2)
        self.iterate(loader)

        dataset = GulpImageDataset(self.temp_dir)
        self.iterate(loader)
