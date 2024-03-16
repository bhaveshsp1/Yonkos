import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
# Load dataset
diction={}
def load_dataset(folder):
    images = []
    labels = []
    for label, person_name in enumerate(os.listdir(folder)):
        diction[label]=person_name
        person_folder = os.path.join(folder, person_name)
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(255, 255))
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
            labels.append(label)  # Use folder name as label
    return np.array(images), np.array(labels)

# Define CNN architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Load dataset
dataset_folder = "dataset"
images, labels = load_dataset(dataset_folder)

# Preprocess images
images = images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

# Split dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
num_classes = len(np.unique(labels))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define model
model = create_model(input_shape=(255, 255, 3), num_classes=num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# Train model
batch_size = 16
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=30,  # You may need to adjust the number of epochs based on your dataset and model performance
    validation_data=val_generator,
    validation_steps=len(X_val) // batch_size
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save model
model.save("face_recognition_model.keras")




# File path to save the dictionary
file_path = "data.json"

# Write the dictionary to a JSON file
with open(file_path, "w") as file:
    json.dump(diction, file)