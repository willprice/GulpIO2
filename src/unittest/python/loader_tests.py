import os
import unittest
import numpy as np
from gulpio.loader import DataLoader
from gulpio.dataset import GulpVideoDataset


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
    # TODO: add sample gulpio files for testing

    def iterate(self, loader):
        for idx, data in enumerate(loader):
            assert len(data) == 2
            if idx == 5:
                break

    def test_dataset(self):
        if os.path.exists('/data/20bn-gestures/gulpio'):
            dataset = GulpVideoDataset('/data/20bn-gestures/gulpio', 2, 2,
                                       False)
            loader = DataLoader(dataset, num_workers=0)
            self.iterate(loader)

            loader = DataLoader(dataset, num_workers=2)
            self.iterate(loader)

            dataset = GulpVideoDataset('/data/20bn-gestures/gulpio', 2, 2,
                                       False, stack=False)
            self.iterate(loader)
        else:
            pass
