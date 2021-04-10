from .face_identifier import FaceIdentifier
from utils.common import unite_dicts
from utils.image_processing import pad_image, resize_image
import numpy as np
import torch
import logging




class LiveFaceIdentifier(FaceIdentifier):

    def __init__(self, known_faces_dir, liveness_detector):
        super(LiveFaceIdentifier, self).__init__(known_faces_dir)
        self.liveness_detector = liveness_detector

    def identify(self, images):
        cache = {}
        results = []
        logging.info('Detecting faces...')
        for im in images:
            hits = super().identify(im)
            cache = unite_dicts(hits, cache)

        logging.info(f'Detected {len(cache.values())} faces.')
        known_faces = {k:v for k,v in cache.items() if k != 'unknown'}
        logging.info(f'Detected {len(cache.keys())} persons. {len(known_faces)} are known.')
        for name, faces in known_faces.items():
            if len(faces) >= 3:
                logging.info(f'Checking liveness of {name}')
                faces = np.asarray(faces)[np.random.choice(len(faces), 3, replace=False)]
                locs, frames = zip(*faces)
                input = self.liveness_detector.preprocess(frames).cuda()
                result = torch.sigmoid(self.liveness_detector(input).squeeze())
                results.append({'name': name, 'alive_conf': float(result.detach().cpu().numpy())})
            else:
                results.append({'name': name, 'message': 'Not enough faces found', 'details': f'Min faces {3}. Found {len(faces)}'})
                logging.info(f'Not enough faces for {name}')
        return results
