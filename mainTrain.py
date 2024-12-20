import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


image_directory = 'datasets/'

# List all images in the "no" and "yes" folders (no tumor and tumor images respectively)
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')
dataset = [] # To store all image data as NumPy arrays.
label = [] # To store corresponding labels (0 = no tumor, 1 = tumor)
INPUT_SIZE=64 # Input image size expected by the model.

# Load and preprocess all "no tumor" images.
for i, image_name in enumerate(no_tumor_images):
    if image_name.lower().endswith('.jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name)
        if image is None: # Skip files that couldn't be read.
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV default) to RGB.
        image = Image.fromarray(image, 'RGB')          # Convert to PIL Image for further processing.
        image = image.resize((INPUT_SIZE, INPUT_SIZE)) # Resize image to 64x64 pixels.
        dataset.append(np.array(image))                # Add the image data as a NumPy array to the dataset.
        label.append(0)                                # Add label 0 (no tumor).

# Load and preprocess all "yes tumor" images.
for i, image_name in enumerate(yes_tumor_images):
    if image_name.lower().endswith('.jpg'):
        image = cv2.imread(image_directory + 'yes/' + image_name)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1) # Add label 1 (tumor).

# Convert the dataset and labels lists to NumPy arrays for processing.
dataset=np.array(dataset)
label=np.array(label)

# Split the data into training and testing sets (80% train, 20% test).
x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize image data to range [0, 1] (helps training convergence).
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# Convert labels to one-hot encoded format (e.g., 0 -> [1, 0], 1 -> [0, 1])
y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)

# Building the CNN model.
model=Sequential()

# Add first convolutional layer: 32 filters, 3x3 kernel, ReLU activation.
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Down-sample using max pooling.
 
# Add second convolutional layer: 32 filters, 3x3 kernel, ReLU activation.
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform')) # He initializer for better weight initialization.
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Add third convolutional layer: 32 filters, 3x3 kernel, ReLU activation.
model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the 3D feature maps to a 1D vector.
model.add(Flatten())

# Add a fully connected dense layer with 64 neurons and ReLU activation.
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5)) # Dropout to prevent overfitting.

# Add the output layer with 2 neurons (2 classes) and softmax activation.
model.add(Dense(2))
model.add(Activation('softmax'))

# Compile the model with categorical crossentropy loss and Adam optimizer.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data.
model.fit(x_train, y_train, batch_size=16, verbose=True, epochs=10, validation_data=(x_test, y_test), shuffle=False)

# Save the trained model to a file for later use.
model.save('BrainTumor10EpochsCategorical.h5')

