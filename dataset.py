# DataLoader and Data-Transform functions are in this 'dataset.py' file

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import os
import random

# DataLoader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])).astype(np.float32)
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])).astype(np.float32)

        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data 
    

## Data-Transform function

# From Numpy --> Pytorch Tensor
class ToTensor(object):
    def __call__(self, data):
        label = data['label']
        input = data['input']
        
        # Image in Numpy = [Y,X,Channel]
        # Image in Pytoch Tensor = [Channel, Y, X]
        # So we need to transpose the axis

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        
        # Making as a dictionary
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
# Data Normalization
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        
        # Only Normalization applied to input (NOT label)
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# Random Augmentation
class RandomFlip(object):
    def __call__(self, data):
        label = data['label']
        input = data['input']
        
        # 50% left/right flip
        if np.random.rand() > 0.5:
            label = np.fliplr(label) 
            input = np.fliplr(input)
        # 50% up/down flip
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data