import os

import cv2
import face_recognition
from utils.utils import make_dir_if_needed

input_path = '../../data/dima/spoofed.MOV'
output_dir = '../../data/dima/spoofed'

make_dir_if_needed(output_dir)

video_capture = cv2.VideoCapture(input_path)

process_this_frame = True
i = 0
while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)

    for j, (top, right, bottom, left) in enumerate(face_locations):
        face_frame = frame[top:bottom, left:right]
        cv2.imwrite(os.path.join(output_dir, '{:04d}_{}.jpg'.format(i, j)), face_frame)
    i += 1

video_capture.release()
