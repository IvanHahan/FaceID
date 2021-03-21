import cv2
from utils.face_identifier import FaceIdentifier
from utils.common import unite_dicts
from liveness_detection.sequence.liveness_detector import LivenessDetector
from utils.image_processing import resize_image, pad_image
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

face_identifier = FaceIdentifier('data/known_faces')
liveness_detector = LivenessDetector.load_from_checkpoint('models/48d39aa512e74102890e7f56c5e3100a/artifacts/model/data/model.pth', map_location='cpu')

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
            frames = liveness_detector.preprocess(frames)
            result = liveness_detector(frames)
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