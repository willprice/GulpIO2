#!/usr/bin/env python
import os
import json

import pandas as pd

from gulpio.utils import (get_single_video_path,
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

    def __init__(self, json_file, folder, frame_size=-1, shm_dir_path='/dev/shm'):
        self.folder = folder
        self.shm_dir_path = shm_dir_path
        self.data = self.read_json(json_file)
        self.frame_size = frame_size

    def read_json(self, json_file):
        with open(json_file, 'r') as f:
            content = json.load(f)
        return content

    def get_meta(self):
        return [{'id': entry['id'], 'label': entry['template']}
                for entry in self.data]

    def iter_data(self):
        for meta in self.get_meta():
            video_folder = os.path.join(self.folder, str(meta['id']))
            video_path = get_single_video_path(video_folder)
            frame_paths = burst_video_into_frames(video_path,
                                                  self.shm_dir_path)
            frames = resize_images(frame_paths, self.frame_size)
            result = {'meta': meta,
                      'frames': frames,
                      'id': meta['id']}
            yield result


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


