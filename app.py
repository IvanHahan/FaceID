import os

from flasgger import Swagger
from flask import Flask

from flask import jsonify, request, abort
import cv2
import os
from utils.live_face_identifier import LiveFaceIdentifier
from liveness_detection.sequence.liveness_detector import LivenessDetector
import torch
import os
import logging
import random
import string
import shutil

logging.getLogger().setLevel(logging.INFO)

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', 'uploads/')
MODEL_PATH = os.environ.get('LIVENESS_DETECTOR_PATH', 'model/liveness_detector.ckpt')
KNOWN_FACES_DIR = os.environ.get('KNOWN_FACES_DIR', 'known_faces/')
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


@app.route('/face_id', methods=['POST', 'DELETE'])
def face_id():
    """
        Endpoint identifying faces in query photos and estimate their liveness.
        ---
        responses:
            200:
                description: Each object in the list represents the identified face (one of known faces). If no faces identified or faces are unknown, list will be empty
                schema:
                    type: array
                    items:
                        oneOf:
                            - type: object
                              properties:
                                name:
                                    type: string
                                alive_conf:
                                    type: number
                            - type: object
                              properties:
                                name:
                                    type: string
                                message:
                                    type: string
                                detail:
                                    type: string
                              required:
                                - name
                                - message

        """
    class_ = request.json.get('ent')
    if class_ is None:
        return {"success": False, "message": "Class not found"}
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
        face_identifier = LiveFaceIdentifier(os.path.join(KNOWN_FACES_DIR, class_), liveness_detector)
        result = face_identifier.identify(images)
        return jsonify(result)
    elif request.method == 'DELETE':
        reg = request.json.get('id')
        if reg is None:
            return {"success": False, "message": "Id not found"}

        reg_dir = os.path.join(KNOWN_FACES_DIR, class_, reg)
        if os.path.exists(reg_dir):
            shutil.rmtree(reg_dir)
            return {'success': True}
        else:
            return {"success": False, "message": "Id not exist"}


@app.route('/class', methods=['Post', 'Delete'])
def class_():
    if request.method == 'POST':
        alias = request.json.get('alias',
                                 'class_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))
        class_dir = os.path.join(KNOWN_FACES_DIR, alias)

        os.makedirs(class_dir, exist_ok=True)
        return {'alias': alias, "success": True}
    elif request.method == 'DELETE':
        if alias := request.json.get('alias'):
            class_dir = os.path.join(KNOWN_FACES_DIR, alias)
            if len(os.listdir(class_dir)) == 0:
                os.rmdir(class_dir)
                return {"success": True}
            else:
                return {"success": False, "message": "Not empty"}
        else:
            return {"success": False, "message": "Class not found"}


@app.route('/enroll', methods=['Post'])
def enroll():
    class_ = request.form.get('ent')
    if class_ is None:
        return {"success": False, "message": "Class not found"}

    images = []
    for f in request.files.values():
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        image = cv2.imread(path)
        os.remove(path)
        images.append(image)

    face_identifier = LiveFaceIdentifier(os.path.join(KNOWN_FACES_DIR, class_), liveness_detector)
    results = face_identifier.identify(images)

    if len(results) > 0:
        reg = results[0]['ent']
        path = os.path.join(KNOWN_FACES_DIR, class_, reg,
                            'image_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png')
        cv2.imwrite(path, image)
        return {'success': False, 'message': 'The person already enrolled', 'name': reg, 'image_path': path}
    else:
        reg = 'reg_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        reg_dir = os.path.join(KNOWN_FACES_DIR, class_, reg)
        os.makedirs(reg_dir, exist_ok=True)
        image_path = os.path.join(reg_dir,  'image_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png')
        cv2.imwrite(image_path, image)
        return {'success': True, 'name': reg}


@app.route('/verify', methods=['Post'])
def verify():
    class_ = request.form.get('ent')
    reg = request.form.get('id')
    if class_ is None:
        return {"success": False, "message": "Class not found"}
    if reg is None:
        return {"success": False, "message": "Id not found"}
    images = []
    for f in request.files.values():
        path = os.path.join(KNOWN_FACES_DIR, f.filename)
        f.save(path)
        image = cv2.imread(path)
        os.remove(path)
        images.append(image)

    face_identifier = LiveFaceIdentifier(os.path.join(KNOWN_FACES_DIR, class_, reg), liveness_detector)
    result = face_identifier.identify(images)
    return jsonify(result)


@app.route('/status', methods=['Get'])
def status():
    return 'ok'


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers.extend(gunicorn_logger.handlers)
    app.logger.setLevel(gunicorn_logger.level)


if __name__ == '__main__':
    app.run(debug=True)


