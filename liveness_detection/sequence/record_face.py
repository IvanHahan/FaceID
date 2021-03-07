import cv2
import os
from utils.common import make_dir_if_needed
import face_recognition

# 1.creating a video object
video = cv2.VideoCapture(0)
# 2. Variable
a = 0

output_dir = '../../data/spoofed_photos/67/'

make_dir_if_needed(output_dir)

# 3. While loop

i = 0
while True:
    # Grab a single frame of video
    ret, frame = video.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)

    for location in face_locations:
        top, right, bottom, left = location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.imwrite(os.path.join(output_dir, '{:04d}.jpg'.format(i)), frame[top:bottom, left:right])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        i += 1

    cv2.imshow('Video', frame)

  # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()

