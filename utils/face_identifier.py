import face_recognition
import cv2
import numpy as np
from glob import glob
from os.path import join
import os
import random


def extract_faces(image):
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    return face_encodings, np.array(face_locations) * 4


def extract_face_series(hits, series_len=5):
    most_probable_faces_i = np.argmax([len(f) for f in hits.values()])
    face_name = list(hits.keys())[most_probable_faces_i]
    faces = list(hits.values())[most_probable_faces_i]
    assert len(faces) >= series_len
    faces = np.asarray(random.sample(faces, series_len))
    return faces, face_name


class FaceIdentifier:

    def __init__(self, known_faces_dir):
        self._known_face_names = []
        self._known_face_encodings = []
        for path in glob(join(known_faces_dir, '**', '*.png'), recursive=True):
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                continue
            self._known_face_encodings.append(encodings[0])
            self._known_face_names.append(os.path.basename(os.path.dirname(path)))

    def identify(self, image):
        face_encodings, face_locations = extract_faces(image)

        hits = {}
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self._known_face_encodings, encoding, tolerance=0.5)

            face_distances = face_recognition.face_distance(self._known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            top, right, bottom, left = location
            face_frame = image[top:bottom, left:right]
            if matches[best_match_index]:
                name = self._known_face_names[best_match_index]
                hits[name] = [(location, face_frame)]
            else:
                hits['unknown'] = [(location, face_frame)]

        return hits



