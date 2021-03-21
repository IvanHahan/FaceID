import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from liveness_detection.sequence.augmentation import ToFlow
from sklearn.feature_extraction.image import extract_patches_2d


def patches_from_image(image, num_patches=5):
    patches = []
    for i in range(num_patches):
        # placeholder = np.zeros((128, 128, 3))
        w = np.random.randint(6, min(224, min(image.shape[:2])), 1)[0]
        x = np.random.randint(0, image.shape[1] - w, 1)[0]
        y = np.random.randint(0, image.shape[0] - w, 1)[0]
        # placeholder[:w, :w, :] = image[y:y+w, x:x+w, :]
        patches.append(image[y:y+w, x:x+w, :])
    return np.asarray(patches)


class LivenessDataset(torch.utils.data.Dataset):

    def __init__(self, live_dirs_file, spoofed_dirs_file, transform, series_len):
        super().__init__()
        self.image_paths = []
        self.label = []

        with open(live_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                series_ = np.sort(np.array(glob.glob(os.path.join('/home/ihahanov/Projects/FaceID/', dir, '*'))))
                if len(series_) == 0:
                    raise FileNotFoundError(dir)
                label_ = np.ones((len(series_), 1))
                self.image_paths.extend(series_)
                self.label.extend(label_)

        with open(spoofed_dirs_file) as f:
            for dir in f.readlines():
                dir = dir.strip()
                series_ = np.array(glob.glob(os.path.join('/home/ihahanov/Projects/FaceID/', dir, '*')))
                if len(series_) == 0:
                    raise FileNotFoundError(dir)
                label_ = np.zeros((len(series_), 1))
                self.image_paths.extend(series_)
                self.label.extend(label_)

        self.image_paths = np.asarray(self.image_paths)
        self.label = np.asarray(self.label)

        permutation = np.random.permutation(len(self.image_paths))
        self.image_paths = self.image_paths[permutation]
        self.label = self.label[permutation]
        self.transform = transform
        self.to_flow = ToFlow()

    def __getitem__(self, item):
        im_path = self.image_paths[item]
        label = self.label[item]
        image = cv2.imread(im_path)
        patches = patches_from_image(image, 5)
        patches = [self.transform(p) for p in patches]
        return patches, [label for _ in range(5)]

    def __len__(self):
        return len(self.image_paths)


