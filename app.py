import os

from flasgger import Swagger
from flask import Flask

from flask import jsonify, request, send_from_directory, abort
import cv2
import os
from utils.live_face_identifier import LiveFaceIdentifier
from liveness_detection.liveness_detector import LivenessDetector

STATIC_DIR = os.environ.get('STATIC', 'static')
MODEL_PATH = os.environ.get('MODEL_PATH')
KNOWN_FACES_DIR = os.environ.get('KNOWN_FACES_DIR')
CONFIG = os.environ.get('CONFIG')


UPLOAD_DIR = os.path.join(STATIC_DIR, 'img')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
conf_object = os.path.join('config.{}'.format(CONFIG))
app.config.from_object(conf_object)
app.config['SECRET_KEY'] = b'lfgp;lhfp;l,mgh;lfl,'

swagger = Swagger(app)
liveness_detector = LivenessDetector.load_from_checkpoint(MODEL_PATH, map_location='cpu')
face_identifier = LiveFaceIdentifier(KNOWN_FACES_DIR, liveness_detector)


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


@app.route('/face_id', methods=['Post'])
def face_id():
    images = []
    for f in request.files.values():
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        image = cv2.imread(path)
        images.append(image)
    result = face_identifier.identify(images)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


