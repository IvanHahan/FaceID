from .face_identifier import FaceIdentifier
from utils.common import unite_dicts
from utils.image_processing import pad_image, resize_image
import numpy as np
import torch


class LiveFaceIdentifier(FaceIdentifier):

    def __init__(self, known_faces_dir, liveness_detector):
        super(LiveFaceIdentifier, self).__init__(known_faces_dir)
        self.liveness_detector = liveness_detector

    def identify(self, images):
        cache = {}
        results = []
        for im in images:
            hits = super().identify(im)
            cache = unite_dicts(hits, cache)
        for name, faces in cache.items():
            if len(faces) >= 5:
                faces = np.asarray(faces)[np.random.choice(len(faces), 5)]
                locs, frames = zip(*faces)
                input = self.liveness_detector.preprocess(frames)
                result = torch.sigmoid(self.liveness_detector(input).squeeze())
                results.append({'name': name, 'alive_conf': float(result.detach().cpu().numpy())})
        return results
