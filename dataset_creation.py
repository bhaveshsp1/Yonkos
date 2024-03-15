import os
import cv2
import numpy as np
from tkinter import Tk, Button, Label, Entry, messagebox
from PIL import Image, ImageTk


class SmartAttendanceSystem:
    def __init__(self):
        self.root = Tk()
        self.root.title("Smart Attendance System")
        self.capture = cv2.VideoCapture(0)

        self.name_label = Label(self.root, text="Enter Name:")
        self.name_label.pack()
        self.name_entry = Entry(self.root)
        self.name_entry.pack()

        self.capture_button = Button(self.root, text="Capture Images", command=self.capture_images)
        self.capture_button.pack()

        self.exit_button = Button(self.root, text="Exit", command=self.exit)
        self.exit_button.pack()

        self.root.mainloop()

    def capture_images(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        images_folder = "dataset/"+name
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        count = 0
        while count < 100:
            ret, frame = self.capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cv2.imwrite(f"{images_folder}/{name}_{count}.jpg", roi_gray)
                count += 1
                cv2.putText(frame, f"Captures: {count}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.waitKey(100)
            cv2.imshow('Capturing Images', frame)
            cv2.waitKey(1)
            if count == 100:
                break

        messagebox.showinfo("Success", f"Images for {name} captured successfully")


    def exit(self):
        self.capture.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    app = SmartAttendanceSystem()

