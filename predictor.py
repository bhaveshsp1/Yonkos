import cv2
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model('mangal_recognition_model.h5')  # Load your trained model


# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Use the default camera (usually webcam)

    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        cv2.imwrite('captured_image.jpg', frame)  # Save the captured image
        cap.release()  # Release the camera
        predict_person('captured_image.jpg')  # Predict the person in the captured image
    else:
        messagebox.showerror("Error", "Failed to capture image")


# Function to predict the person in the image
def predict_person(image_path):
    img = image.load_img(image_path, target_size=(196, 196))  # Resize image to match model input size
    img_array = image.img_to_array(img) / 255.0  # Convert image to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape

    # Predict the person using the model
    prediction = model.predict(img_array)[0][0]  # Assuming binary classification, adjust as needed

    # Display the prediction
    if prediction >= 0.2:  # Assuming threshold for positive prediction is 0.5, adjust as needed
        messagebox.showinfo("Prediction", "The person is Dheva")  # Replace Dheva with the predicted name
    else:
        messagebox.showinfo("Prediction", "The person is not Dheva")  # Replace Dheva with the predicted name


# Create the GUI window
root = tk.Tk()
root.title("Face Recognition")

# Create buttons
capture_button = tk.Button(root, text="Capture", command=capture_image)
capture_button.pack()

# Start the Tkinter event loop
root.mainloop()
