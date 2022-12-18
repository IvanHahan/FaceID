import os

from flasgger import Swagger
from flask import Flask

from flask import jsonify, request, send_from_directory, abort
import cv2
import os
from utils.live_face_identifier import LiveFaceIdentifier
from liveness_detection.liveness_detector import LivenessDetector
import random
import string

STATIC_DIR = os.environ.get('STATIC', 'static')
MODEL_PATH = os.environ.get('MODEL_PATH')
KNOWN_FACES_DIR = os.environ.get('KNOWN_FACES_DIR')
CONFIG = os.environ.get('CONFIG')

os.makedirs(STATIC_DIR, exist_ok=True)

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
        path = os.path.join(STATIC_DIR, f.filename)
        f.save(path)
        image = cv2.imread(path)
        images.append(image)
    result = face_identifier.identify(images)
    return jsonify(result)


@app.route('/class', methods=['Post', 'Delete'])
def class_():
    if request.method == 'POST':
        alias = request.json.get('alias',
                                 'class_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)))
        class_dir = os.path.join(STATIC_DIR, alias)

        os.makedirs(class_dir, exist_ok=True)
        return {'alias': alias}
    elif request.method == 'DELETE':
        if alias := request.json.get('alias'):
            class_dir = os.path.join(STATIC_DIR, alias)
            if len(os.listdir(class_dir)) == 0:
                os.rmdir(class_dir)
                return {"success": True}
            else:
                return {"success": False, "message": "Not empty"}
        else:
            return {"success": False, "message": "Class not found"}


@app.route('/enroll', methods=['Post'])
def enroll():
    class_ = request.json.get('ent')
    image_file = next(request.files.values(), None)
    if class_ is None:
        return {"success": False, "message": "Class not found"}
    if image_file is None:
        return {"success": False, "message": "Image not found"}

    path = os.path.join(STATIC_DIR, image_file.filename)
    image_file.save(path)
    image = cv2.imread(path)
    results = face_identifier.identify([image])

    if len(results) > 0:
        reg = results[0]['ent']
        path = os.path.join(STATIC_DIR, class_, reg,
                            'image_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png')
        cv2.imwrite(path, image)
        return {'success': False, 'message': 'The person already enrolled'}
    else:
        reg = 'reg_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        reg_dir = os.path.join(STATIC_DIR, class_, reg)
        os.makedirs(reg_dir, exist_ok=True)
        image_path = os.path.join(reg_dir,  'image_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png')
        cv2.imwrite(image_path, image)
        return {'success': True, 'name': reg}








if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


