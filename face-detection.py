import numpy as np
import cv2

# Load the cascade (file upon which the face detection is based) taken from openCV
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_alt2.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(1)

# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while (True):
    # capture frame by frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y +  h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display
    cv2.imshow('img', frame)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()