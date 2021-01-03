import cv2
import torch
import argparse
from glob import glob
from os.path import join
from utils.face_identifier import FaceIdentifier, extract_face_series
from utils.common import unite_dicts
from liveness_detection.util import preprocess_input as liveness_process
from utils import visualization
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', help='directory storing registered users', default='data/16.11.20/reg_1')
parser.add_argument('--test_dir', help='directory storing test users', default='data/16.11.20/test_1')
parser.add_argument('--weights', help='trained weight path', default='model/efficientnet-b0-best.pth')
parser.add_argument('--liveness_series_len', help='Num of image for liveness detection', default=5, type=int)


args = parser.parse_args()

# Identify face
face_identifier = FaceIdentifier(args.sample_dir)
hits = []
for i, path in enumerate(glob(join(args.test_dir, '*.png'))):
    image = cv2.imread(path)

    hits.append(face_identifier.identify(image))

hits = unite_dicts(*hits)

# Extract identified face series
face_series, name = extract_face_series(hits, 5)

visualization.visualize_series(face_series)

# Load liveness detection model
model = mlflow.pytorch.load_model('liveness_detection/mlruns/0/a09fba597d5f4ecd9c745c054b2d24c9/artifacts/model')
# model = LivenessDetector.load_from_checkpoint('liveness_detection/mlruns/0/a09fba597d5f4ecd9c745c054b2d24c9/artifacts/model/data/model.pth')
model.eval()
# model.load_state_dict(torch.load(args.weights, map_location='cpu'))

# Perform liveness detection
liveness_image = liveness_process(face_series).to('cuda')
result = torch.sigmoid(model(liveness_image)).cpu().detach().numpy().squeeze()

print(name, result)


