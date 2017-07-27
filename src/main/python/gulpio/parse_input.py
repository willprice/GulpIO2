#!/usr/bin/env python
import os

from tbntools import data
import pandas as pd

from gulpio.utils import (find_images_in_folder,
                          get_video_path,
                          resize_images,
                          burst_video_into_frames)


class MetaDataIterator(object):

    def iter_data():
        """ ({meta: dict with meta information,
              files: [frames or mp3 with path]},
              id: unique id to find the video again
             {...}
             ...)
        """
        return NotImplementedError

    def __getitem__(self, i):
        return NotImplementedError


class Input_from_json(object):

    def __init__(self, json_file, folder, shm_dir_path='/dev/shm'):
        self.folder = folder
        self.shm_dir_path = shm_dir_path
        self.data = self.read_json_file(json_file)
        self.labels2idx = self.create_labels_dict()

    def read_json_file(self, json_file):
        return data.RawDataset.load(json_file, label='template').storage

    def extract_labels(self, key='template'):
        labels = []
        for item in self.data:
            if key in item:
                labels.append(item[key])
        labels = sorted(set(labels))
        return labels

    def create_labels_dict(self):
        labels = self.extract_labels()
        labels2idx = {}
        for i, label in enumerate(labels):
            labels2idx[label] = i
        return labels2idx

    def get_data(self):
        output = []
        for entry in self.data:
            row = {}
            row['start_time'] = None
            row['end_time'] = None
            row['id'] = entry['id_']
            row['label'] = entry['template']
            output.append(row)
        return output, self.labels2idx

    def iter_data(self):
        meta_data, _ = self.get_data()
        sub_folders = (os.path.join(self.folder, md['id']) for md in meta_data)
        return iter(({'meta': md,
                      'frames':
                      resize_images(burst_video_into_frames(get_video_path(sub_folder)[0],
                                                                      self.shm_dir_path)
                                ),
                      'id': md['id']}
                    for md, sub_folder in zip(meta_data, sub_folders)))


class Input_from_csv(object):

    def __init__(self, csv_file, num_labels=None):
        self.num_labels = num_labels
        self.data = self.read_input_from_csv(csv_file)
        self.labels2idx = self.create_labels_dict()

    def read_input_from_csv(self, csv_file):
        print(" > Reading data list (csv)")
        return pd.read_csv(csv_file)

    def create_labels_dict(self):
        labels = sorted(pd.unique(self.data['label']))
        if self.num_labels:
            assert len(labels) == self.num_labels
        labels2idx = {}
        for i, label in enumerate(labels):
            labels2idx[label] = i
        return labels2idx

    def get_data(self):
        output = []
        for idx, row in self.data.iterrows():
            entry_dict = {}
            entry_dict['id'] = row.youtube_id
            entry_dict['label'] = row.label
            entry_dict['start_time'] = row.time_start
            entry_dict['end_time'] = row.time_end
            output.append(entry_dict)
        return output, self.labels2idx


