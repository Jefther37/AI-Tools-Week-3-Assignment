
# Buggy TensorFlow Script - Troubleshooting Challenge

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# BUG 1: Incorrect input shape for Conv2D - should be 4D
# BUG 2: Labels not one-hot encoded
# BUG 3: Loss function mismatch (sparse_categorical_crossentropy expected for integer labels, but using categorical labels)
# BUG 4: Missing activation in Dense layer

model = Sequential([
    Conv2D(32, (3, 3), input_shape=(28, 28), activation='relu'),  # Bug: input_shape missing channel dimension
    Flatten(),
    Dense(128),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Should match label format
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)  # y_train is not one-hot encoded
