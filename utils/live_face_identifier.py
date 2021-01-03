from .face_identifier import FaceIdentifier
from utils.common import unite_dicts
from utils.image_processing import pad_image, resize_image
import numpy as np


class LiveFaceIdentifier(FaceIdentifier):

    def __init__(self, known_faces_dir, liveness_detector):
        super(LiveFaceIdentifier, self).__init__(known_faces_dir)
        self.liveness_detector = liveness_detector

    def identify(self, images):
        cache = {}
        for im in images:
            hits = super().identify(im)
            cache = unite_dicts(hits, cache)
        for name, faces in cache.items():
            if len(faces) >= 5:
                faces = np.asarray(faces)[np.random.choice(len(faces), 5)]
                locs, frames = zip(*faces)
                frames = np.array([pad_image(resize_image(f, 128), (128, 128))[0] for f in frames], dtype='uint8')
                result = self.liveness_detector.predict(frames)[0] > 0.95
                cache[name] = result
        return cache