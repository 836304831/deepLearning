# -*- coding:utf-8 -*-
# ----------------------
# File: data_manager.py
# Author: acedar
# Date: 2019.03.01
# ----------------------

import os
import pickle
import numpy as np


__dir__ = os.path.abspath(os.path.dirname(__file__))

WORK_DIR = os.path.dirname(os.path.dirname(__dir__))
cifar10_path = os.path.join(WORK_DIR, "datasets/cifar-10-batches-py")

train_file_list = [os.path.join(cifar10_path, "data_batch_1"), os.path.join(cifar10_path, "data_batch_2"),
                   os.path.join(cifar10_path, "data_batch_3"), os.path.join(cifar10_path, "data_batch_4"),
                   os.path.join(cifar10_path, "data_batch_5")]

test_file_list = [os.path.join(cifar10_path, "test_batch")]


class DataManager(object):
    """ for cifar-10 data"""
    def __init__(self, data_file_list, need_shuffle=False):
        self.data_file_list = data_file_list
        self.all_data = []
        self.all_labels = []
        self.index = 0
        self.need_shuffle = need_shuffle

    def load_data(self):
        print("will read file num: {}".format(len(self.data_file_list)))
        for file in self.data_file_list:
            with open(file, "rb") as f:
                loaded_data = pickle.load(f, encoding="ISO-8859-1")
                data = loaded_data.get("data")
                labels = loaded_data.get("labels")
                print("load_file: {}, data.shape: {}， labels.len: {}".format(file, data.shape, len(labels)))
                self.all_data.append(data)
                self.all_labels.append(labels)
        self.all_data = np.vstack(self.all_data)
        self.all_labels = np.hstack(self.all_labels)
        self.all_data = self.all_data / 127.5 - 1
        # self.all_data = self.all_data / 255
        print("data type:{}".format(type(self.all_data)))
        print("all_data.shape: {}, all_labels: {}".format(self.all_data.shape, self.all_labels.shape))
        return self.all_data, self.all_labels

    def filter_binary_data(self):

        tmp_data, tmp_label = [], []
        for row, label in zip(self.all_data, self.all_labels):
            if label == 0 or label == 1:
                tmp_data.append(row)
                tmp_label.append(label)

        self.all_data = tmp_data
        self.all_labels = tmp_label
        self.all_data = np.vstack(self.all_data)
        self.all_labels = np.hstack(self.all_labels)
        print("filter data all_data.shape:{}, all_labels.shape:{}".format(self.all_data.shape, self.all_labels.shape))
        return self.all_data, self.all_labels

    def shuffle_data(self):
        idx = np.random.permutation(self.all_data.shape[0])
        self.all_data = self.all_data[idx]
        self.all_labels = self.all_labels[idx]

    def next_batch(self, batch_size):
        if batch_size > self.all_data.shape[0]:
            return self.all_data, self.all_labels

        if self.index + batch_size > self.all_data.shape[0]:
            if self.need_shuffle:
                self.shuffle_data()
                self.index = 0
            else:
                raise Exception("no more data to return")

        batch_data = self.all_data[self.index: self.index + batch_size]
        batch_label = self.all_labels[self.index: self.index + batch_size]
        self.index += batch_size
        return batch_data, batch_label


if __name__ == "__main__":
    data_manager = DataManager(test_file_list)
    data_manager.load_data()
    data_manager.next_batch(32)
