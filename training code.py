import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Set dataset paths
dataset_path = "C:\\Users\\LOQ\\OneDrive\\Desktop\\archive (2)\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(dataset_path, 'train')
valid_dir = os.path.join(dataset_path, 'valid')





print("Train directory:", train_dir)
print("Valid directory:", valid_dir)
print("Exists?", os.path.exists(train_dir), os.path.exists(valid_dir))

img_size = (128, 128)
batch_size = 32
epochs = 15

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print("Detected Classes: ", class_names)

# Normalize data
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y))

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=img_size+(3,)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train and save the model
model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs)
model.save('C:\\Users\\LOQ\\OneDrive\\Desktop\\dataset\\model.h5')
