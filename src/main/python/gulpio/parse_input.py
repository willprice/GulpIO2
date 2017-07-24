#!/usr/bin/env python

from tbntools import data
import pandas as pd


class Input_from_json(object):

    def __init__(self, json_file):
        self.data = self.read_json_file(json_file)
        self.label2idx = self.create_labels_dict()

    def read_json_file(self, json_file):
        return data.RawDataset.load(json_file, label='template').storage

    def create_labels_dict(self, key='template'):
        labels = sorted(set([item[key] for item in self.data]))
        labels2idx = {}
        for i, label in enumerate(labels):
            labels2idx[label] = i
        return labels2idx

    def get_data(self):
        output = []
        for entry in self.data:
            entry['start_time'] = None
            entry['end_time'] = None
            output.append(entry)
        return output


class Input_from_csv(object):

    def __init__(self, csv_file, num_labels=None):
        self.num_labels = num_labels
        self.data = self.read_input_from_csv(csv_file)
        self.label2idx = self.create_labels_dict()

    def read_input_from_csv(self, csv_file):
        print(" > Reading data list (csv)")
        return pd.read_csv(csv_file)

    def create_labels_dict(self):
        labels = sorted(pd.unique(df['label']))
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
        return output


