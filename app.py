from flasgger import Swagger
from flask import Flask

from flask import jsonify, request, abort
import cv2
import os
from utils.live_face_identifier import LiveFaceIdentifier
from liveness_detection.sequence.liveness_detector import LivenessDetector
import logging
import torch
import os

logging.getLogger().setLevel(logging.INFO)

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', 'uploads/')
MODEL_PATH = os.environ.get('LIVENESS_DETECTOR_PATH', '/home/ihahanov/Projects/FaceID/model/liveness_detector.ckpt')
KNOWN_FACES_DIR = os.environ.get('KNOWN_FACES_DIR', '/home/ihahanov/Projects/FaceID/data/16.11.20/')
CONFIG = os.environ.get('CONFIG', 'Default')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


app = Flask(__name__)
conf_object = os.path.join('config.{}'.format(CONFIG))
app.config.from_object(conf_object)
app.config['SECRET_KEY'] = b'lfgp;lhfp;l,mgh;lfl,'

swagger = Swagger(app)
liveness_detector = LivenessDetector()
liveness_detector.load_state_dict(torch.load(MODEL_PATH))
liveness_detector.eval()
face_identifier = LiveFaceIdentifier(KNOWN_FACES_DIR, liveness_detector.cuda())


class Error(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(Error)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.errorhandler(KeyError)
def handle_key_error(error):
    response = jsonify(Error(str(error)).to_dict())
    response.status_code = error.status_code
    return response


@app.route('/face_id', methods=['POST'])
def face_id():
    if request.method == 'POST':
        images = []
        for f in request.files.values():
            if f is None or len(f.filename) == 0:
                continue
            path = os.path.join(UPLOAD_DIR, f.filename)
            f.save(path)
            image = cv2.imread(path)
            os.remove(path)
            images.append(image)
        result = face_identifier.identify(images)
        return jsonify(result)
    abort(404)


if __name__ == '__main__':
    app.run(debug=True)


