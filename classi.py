import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Define paths for training and test datasets
train_data = "D:\Virtual\Deep Learning\Multiclass classification cnn\seg_train"
test_data = "D:\Virtual\Deep Learning\Multiclass classification cnn\seg_test"

# ImageDataGenerator for training set with reduced augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,  # Reduced rotation range
    zoom_range=0.1,     # Reduced zoom range
    width_shift_range=0.1,  # Reduced width shift
    height_shift_range=0.1,  # Reduced height shift
    shear_range=0.1,    # Reduced shear range
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generator for training data
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(100, 100),  # Reduced image size
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    shuffle=True
)

# ImageDataGenerator for validation set
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generator for validation data
validation_generator = validation_datagen.flow_from_directory(
    test_data,
    target_size=(100, 100),  # Reduced image size
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    shuffle=False
)

# Create a dictionary for label mappings
labels = {value: key for key, value in train_generator.class_indices.items()}

# Print label mappings
print("Label Mappings for classes present in the training and validation datasets\n")
for key, value in labels.items():
    print(f"{key} : {value}")

# Function to create the CNN model
def create_model():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(100, 100, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Flatten(),
        
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        
        Dense(units=6, activation='softmax')  # Assuming 6 classes
    ])
    
    return model

# Create CNN model
cnn_model = create_model()

# Learning rate scheduler and early stopping callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Compile the model
optimizer = Adam(learning_rate=0.001)
cnn_model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Fit the model
history = cnn_model.fit(
    train_generator,
    epochs=20,  # Reduced number of epochs
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop, checkpoint]
)

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Plot the training history
plot_training_history(history)

# Evaluate the model
validation_generator.reset()  # Reset the generator for accurate evaluation
y_pred = cnn_model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
print("\nConfusion Matrix:\n", confusion_matrix(validation_generator.classes, y_pred_classes))
print("\nClassification Report:\n", classification_report(validation_generator.classes, y_pred_classes, target_names=labels.values()))
