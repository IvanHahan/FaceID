import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from liveness_detection.augmentation import MaxSizeResizer, SquarePad, ToFlow
import glob


class CelebA(torch.utils.data.Dataset):

    def __init__(self, dir):
        super().__init__()
        self.live_im_paths = glob.glob(os.path.join(dir, '*/live/*.png'))
        self.spoof_im_paths = glob.glob(os.path.join(dir, '*/spoof/*.png'))

    def __getitem__(self, item):
        live_face = self.read_image(self.live_im_paths[item])
        spoof_face = self.read_image(self.spoof_im_paths[item])

        return live_face, spoof_face

    def read_image(self, image_path):
        """
        Read an image from input path
        params:
            - image_local_path (str): the path of image.
        return:
            - image: Required image.
        """

        img = cv2.imread(image_path)
        # Get the shape of input image
        real_h, real_w, c = img.shape
        assert os.path.exists(image_path[:-4] + '_BB.txt'), 'path not exists' + ' ' + image_path

        with open(image_path[:-4] + '_BB.txt', 'r') as f:
            material = f.readline()
            try:
                x, y, w, h, score = material.strip().split(' ')
            except:
                print('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')

            try:
                w = int(float(w))
                h = int(float(h))
                x = int(float(x))
                y = int(float(y))
                w = int(w * (real_w / 224))
                h = int(h * (real_h / 224))
                x = int(x * (real_w / 224))
                y = int(y * (real_h / 224))

                # Crop face based on its bounding box
                y1 = 0 if y < 0 else y
                x1 = 0 if x < 0 else x
                y2 = real_h if y1 + h > real_h else y + h
                x2 = real_w if x1 + w > real_w else x + w
                img = img[y1:y2, x1:x2, :]

            except:
                print('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return min(len(self.live_im_paths), len(self.spoof_im_paths))


if __name__ == '__main__':
    dataset = CelebA('/home/ihahanov/Projects/FaceID/data/CelebaSpoof/CelebA_Spoof/Data/test')
    import matplotlib.pyplot as plt
    plt.imshow(dataset[1000][1])
    plt.show()