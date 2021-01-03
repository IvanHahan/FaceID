import cv2
import argparse
import glob
import os
from utils.utils import make_dir_if_needed
import numpy as np
from tqdm import tqdm
from liveness_detection.augmentation import MaxSizeResizer

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', default='data/liza/live')
parser.add_argument('--output_dir', default='data/liza/live_flow')
args = parser.parse_args()


if __name__ == '__main__':
    make_dir_if_needed(args.output_dir)
    files = sorted(glob.glob(os.path.join(args.images_dir, '*.jpg')))

    resize = MaxSizeResizer(224)

    prev_file = files[0]
    for file in tqdm(files[1:]):
        prev_image = cv2.imread(prev_file)
        prev_image = resize(prev_image)
        image = cv2.imread(file)
        image = cv2.resize(image, (prev_image.shape[1], prev_image.shape[0]))

        hsv = np.zeros_like(prev_image)

        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(file)), rgb)
        prev_file = file





