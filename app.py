import os

from flasgger import Swagger
from flask import Flask

from flask import jsonify, request, send_from_directory, abort
import cv2
import mlflow
import os
from utils.live_face_identifier import LiveFaceIdentifier

STATIC_DIR = 'static'
UPLOAD_FOLDER = 'img'
UPLOAD_DIR = os.path.join(STATIC_DIR, UPLOAD_FOLDER)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
swagger = Swagger(app)
liveness_detector = mlflow.pytorch.load_model('model/swa')
face_identifier = LiveFaceIdentifier('data/known_faces', liveness_detector)

conf_object = os.path.join('app.config.{}'.format('Default'))
app.config.from_object(conf_object)
app.config['SECRET_KEY'] = b'lfgp;lhfp;l,mgh;lf,'


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


@app.route('/identify_face', methods=['Post'])
def identify_face():
    images = []
    for f in request.files.values():
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        image = cv2.imread(path)
        images.append(image)
    result = face_identifier.identify(images)
    return jsonify(result)



