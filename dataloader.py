# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:17:40 2020

@author: 18096
"""
from __future__ import division
import numpy as np
import os
import logging
from torch.utils.data import Dataset

logger = logging.getLogger('DeepAR.Data')

class TrainDataset(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f'train_X_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'train_label_{data_name}.npy'))
        self.train_len = self.data.shape[0]
        logger.info(f'Building datasets from {data_path}...')
        logger.info(f'train_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:],self.label[index])

class ValiDataset(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f'vali_X_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'vali_label_{data_name}.npy'))
        self.vali_len = self.data.shape[0]
        logger.info(f'vali_len: {self.vali_len}')
        #logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.vali_len

    def __getitem__(self, index):
        return (self.data[index,:,:],self.label[index])

class TestDataset(Dataset):
    def __init__(self, data_path, data_name):
        self.data = np.load(os.path.join(data_path, f'test_X_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')
        #logger.info(f'building datasets from {data_path}...')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:],self.label[index])

if __name__ == '__main__':
    train_set = TrainDataset(os.path.join('data','Zone1'),
                             'Zone1', 1)