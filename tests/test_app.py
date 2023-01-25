import pytest
import os
from io import BytesIO
import shutil


@pytest.fixture
def test_client():
    os.environ['KNOWN_FACES_DIR'] = 'tests/known_faces'
    from app import app
    return app.test_client()


@pytest.fixture
def new_face():
    return 'tests/new_tom.png'

@pytest.fixture
def old_face():
    return 'tests/old_tom.png'


@pytest.fixture
def known_faces():
    return 'tests/known_faces'


def test_status(test_client):
    res = test_client.get('/status')
    assert res.status_code == 200

def test_enroll_new(test_client, new_face, known_faces):
    with open(new_face, 'rb') as f:
        image_data = BytesIO(f.read())
    class_ = 'toms'
    prev_class_len = len(os.listdir(os.path.join(known_faces, class_)))
    res = test_client.post('/enroll', content_type='multipart/form-data',
                                    data={'image': (image_data, 'img1.jpg'),
                                          'ent': class_})
    new_class_len = len(os.listdir(os.path.join(known_faces, class_)))
    assert new_class_len > prev_class_len
    assert res.status_code == 200
    assert res.json['success'] == True
    shutil.rmtree(os.path.join(known_faces, class_, res.json['name']))


def test_enroll_old(test_client, old_face, known_faces):
    with open(old_face, 'rb') as f:
        image_data = BytesIO(f.read())
    class_ = 'toms'
    res = test_client.post('/enroll', content_type='multipart/form-data',
                                    data={'image': (image_data, 'img1.jpg'),
                                          'ent': class_})
    assert res.json['success'] == False
    assert res.status_code == 200
    os.remove(res.json['image_path'])


def test_verify_success(test_client, old_face):
    class_ = 'toms'
    id = 'holland'
    res = test_client.post('/verify', content_type='multipart/form-data',
                                    data={'images': (open(old_face, 'rb'), 'img1.jpg'),
                                          'image2': (open(old_face, 'rb'), 'img2.jpg'),
                                          'image3': (open(old_face, 'rb'), 'img3.jpg'),
                                          'ent': class_,
                                          'id': id})
    assert len(res.json) > 0
    assert res.status_code == 200


def test_verify_fail(test_client, new_face):
    class_ = 'toms'
    id = 'holland'
    res = test_client.post('/verify', content_type='multipart/form-data',
                                    data={'images': (open(new_face, 'rb'), 'img1.jpg'),
                                          'image2': (open(new_face, 'rb'), 'img2.jpg'),
                                          'image3': (open(new_face, 'rb'), 'img3.jpg'),
                                          'ent': class_,
                                          'id': id})
    assert len(res.json) == 0
    assert res.status_code == 200


def test_class_create(test_client, known_faces):
    res = test_client.post('/class', json={'alias': 'test'})

    assert res.json['success'] == True
    assert res.json['alias'] == 'test'
    assert os.path.exists(os.path.join(known_faces, res.json['alias']))
    shutil.rmtree(os.path.join(known_faces, res.json['alias']))


def test_class_delete(test_client, known_faces):
    res = test_client.post('/class', json={'alias': 'test'})

    assert res.json['success'] == True
    assert res.json['alias'] == 'test'
    assert os.path.exists(os.path.join(known_faces, res.json['alias']))

    alias = res.json['alias']
    res = test_client.delete('/class', json={'alias': 'test'})
    assert not os.path.exists(os.path.join(known_faces, alias))
    assert res.json['success'] == True
