import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from liveness_detection.sequence.augmentation import ToFlow
import random


class SpoofTextureDataset(torch.utils.data.Dataset):

    def __init__(self, live_dirs, spoofed_dirs=None, transform=None, shuffle=True):
        super().__init__()
        self.files = []
        self.label = []
        for dir in live_dirs:
            files = [os.path.join(dir, file) for file in os.listdir(dir)]
            self.files.extend(files)
            self.label.extend(np.ones(len(files)).tolist())

        self.is_training = True
        if spoofed_dirs is not None:
            for dir in spoofed_dirs:
                files = [os.path.join(dir, file) for file in os.listdir(dir)]
                self.files.extend(files)
                self.label.extend(np.zeros(len(files)).tolist())
            self.is_training = False

        self.files = np.array(self.files)
        self.label = np.array(self.label).reshape((-1, 1))

        if shuffle:
            permutation = np.random.permutation(len(self.files))
            self.files = self.files[permutation]
            self.label = self.label[permutation]
        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.files[item])
        label = self.label[item]

        if self.transform:
            image = self.transform(image)

        if self.is_training:
            return image, label
        return image

    def __len__(self):
        return len(self.files)


def list_dir(dir, query):
    return sorted(list(glob.glob(os.path.join(dir, query))))


class LivenessDataset(torch.utils.data.Dataset):

    def __init__(self, live_dirs_file, spoofed_dirs_file, transform, series_len):
        super().__init__()
        self.series = []
        self.label = []
        self.series_len = series_len
        live = []
        spoof = []
        with open(live_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                live.append('/home/ihahanov/Projects/FaceID/' + dir)
        with open(spoofed_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                spoof.append('/home/ihahanov/Projects/FaceID/' + dir)
        self.dirs = np.asarray(live + spoof)
        self.label = np.concatenate([np.ones(len(live)), np.zeros(len(spoof))]).reshape(-1, 1)

        permutation = np.random.permutation(len(self.dirs))
        self.dirs = self.dirs[permutation]
        self.label = self.label[permutation]
        self.transform = transform
        self.to_flow = ToFlow()

    def __getitem__(self, item):
        dir = self.dirs[item]
        files = np.sort(glob.glob(os.path.join(dir, '*')))
        if np.random.uniform(0, 1) > 0.2:
            files = np.random.choice(files, self.series_len, replace=False)
            images = [cv2.imread(file) for file in files]
            label = self.label[item]
        else:
            file = np.random.choice(files, 1, replace=True)[0]
            images = [cv2.imread(file) for _ in range(self.series_len)]
            label = [0]
        # images = self.to_flow(images)
        if self.transform is not None:
            images = self.transform(images)

        return images, label

    def __len__(self):
        return len(self.dirs)


