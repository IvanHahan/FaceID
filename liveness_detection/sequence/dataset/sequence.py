import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from liveness_detection.sequence.augmentation import ToFlow


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

        with open(live_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                series_ = np.sort(np.array(glob.glob(os.path.join('../'+dir, '*'))))
                if len(series_) == 0:
                    raise FileNotFoundError(dir)
                bound = -(series_.shape[0] % series_len)
                series_ = series_ if bound == 0 else series_[:-(series_.shape[0] % series_len)]
                series_ = series_.reshape((-1, series_len))
                label_ = np.ones((len(series_), 1))
                self.series.append(series_)
                self.label.append(label_)

        with open(spoofed_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                series_ = np.array(glob.glob(os.path.join('../'+dir, '*')))
                if len(series_) == 0:
                    raise FileNotFoundError(dir)
                bound = -(series_.shape[0] % series_len)
                series_ = series_ if bound == 0 else series_[:-(series_.shape[0] % series_len)]
                series_ = series_.reshape((-1, series_len))
                label_ = np.zeros((len(series_), 1))
                self.series.append(series_)
                self.label.append(label_)

        self.series = np.row_stack(self.series)
        self.label = np.row_stack(self.label)

        permutation = np.random.permutation(len(self.series))
        self.series = self.series[permutation]
        self.label = self.label[permutation]
        self.transform = transform
        self.to_flow = ToFlow()

    def __getitem__(self, item):
        series = self.series[item]
        label = self.label[item]
        images = [cv2.imread(file) for file in series]
        # images = self.to_flow(images)
        images = self.transform(images)

        return images, label

    def __len__(self):
        return len(self.series)


