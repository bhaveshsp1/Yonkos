import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
batch_size = 32
img_height = 196
img_width = 196
epochs = 20

# Data generator for training
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset',  # Assuming 'dataset' is the directory containing all person folders
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes=['mangal'],  # Only include the folder for "Dheva"
    class_mode='binary')  # Using binary mode since it's just one class

# Load pre-trained CNN model (e.g., VGG16)
base_model = tf.keras.applications.VGG16(
    weights='imagenet',  # Load pre-trained ImageNet weights
    include_top=False,   # Exclude the classification layer
    input_shape=(img_height, img_width, 3))

# Freeze the convolutional base
base_model.trainable = False

# Add custom classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs)

# Save the trained model
model.save('mangal_recognition_model.h5')
