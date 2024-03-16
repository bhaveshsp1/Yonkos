import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained face recognition model
model = load_model("face_recognition_model.h5")

import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the group picture
image = cv2.imread('bc.jpg')  # Replace 'group_picture.jpg' with your image file path

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the resulting image with detected faces
cv2.imshow('Group Picture with Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()