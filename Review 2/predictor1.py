from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import json
# Load pre-trained face recognition model
model = load_model("face_recognition_model.keras")
'''with open('data.json', "r") as file:
    diction = dict(json.load(file))'''
diction={0: 'mangal', 1: 'Vijay'}
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize face region to match model input size (255x255)
        face_roi_resized = cv2.resize(face_roi, (255, 255))

        # Convert grayscale to RGB
        face_roi_rgb = cv2.cvtColor(face_roi_resized, cv2.COLOR_GRAY2RGB)

        # Expand dimensions to make it compatible with model input
        face_roi_rgb = np.expand_dims(face_roi_rgb, axis=0)

        # Normalize pixel values
        face_roi_rgb = face_roi_rgb.astype('float32') / 255.0

        # Predict face label using the model
        pred = model.predict(face_roi_rgb)
        label = np.argmax(pred)

        # Draw bounding box around the face
        color = (255, 0, 0)  # Blue color for bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Write label on the bounding box
        text = f"{diction[label]}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame")
            break

        frame_with_faces = detect_faces(frame)
        cv2.imshow('Face Detection', frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Face Recognition")

button = tk.Button(root, text="Open Camera", command=open_camera)
button.pack()

root.mainloop()