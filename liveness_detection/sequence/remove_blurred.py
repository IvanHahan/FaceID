import cv2
import argparse
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os
from liveness_detection.sequence.augmentation import MaxSizeResizer

parser = argparse.ArgumentParser()
parser.add_argument('--annot_file', default='data/copy.txt')
args = parser.parse_args()


if __name__ == '__main__':
    resizer = MaxSizeResizer(64)
    with open(args.annot_file) as f:
        for dir in tqdm(f.readlines()):
            dir = dir.strip()
            image_paths = np.array(glob.glob(os.path.join(dir, '*')))
            for path in image_paths:
                image = cv2.imread(path)
                image = resizer(image)
                # if cv2.Laplacian(image, cv2.CV_64F).var() < 600:
                #     os.unlink(path)
                print(cv2.Laplacian(image, cv2.CV_64F).var())
                plt.imshow(image)
                plt.show()