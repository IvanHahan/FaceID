import cv2
import torch
from utils.face_identifier import FaceIdentifier
from utils.common import unite_dicts
from liveness_detection.liveness_detection_client import LivenessDetectionClient
from utils.image_processing import resize_image, pad_image
import numpy as np


# def calc_iou(l, r, no_positions=False):
#     # x1, y1, x2, y2
#
#     if no_positions:
#         l = [0, 0, l[2] - l[0], l[3] - l[1]]
#         r = [0, 0, r[2] - r[0], r[3] - r[1]]
#
#     l_x1, l_y1, l_x2, l_y2 = l
#     r_x1, r_y1, r_x2, r_y2 = r
#
#     inter_x1 = max(l_x1, r_x1)
#     inter_y1 = max(l_y1, r_y1)
#     inter_x2 = min(l_x2, r_x2)
#     inter_y2 = min(l_y2, r_y2)
#     intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
#
#     union_x1 = min(l_x1, r_x1)
#     union_y1 = min(l_y1, r_y1)
#     union_x2 = max(l_x2, r_x2)
#     union_y2 = max(l_y2, r_y2)
#     union = (union_x2 - union_x1) * (union_y2 - union_y1)
#
#     return intersection / (union + 0.00001)

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

face_identifier = FaceIdentifier('data/known_faces')
liveness_detector = LivenessDetectionClient()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
cache = {}

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    # if process_this_frame:
    hits = face_identifier.identify(frame)

    cache = unite_dicts(cache, hits)

    # Display the results
    for name, faces in cache.items():
        if len(faces) == 5:
            locs, frames = zip(*faces)
            frames = np.array([pad_image(resize_image(f, 128), (128, 128))[0] for f in frames], dtype='uint8')
            result = liveness_detector.predict(frames)
            color = (0, 255, 0) if result[0] > 0.95 else (0, 0, 255)
            top, right, bottom, left = locs[-1]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cache[name].pop(0)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()