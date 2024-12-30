import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

class DataPreprocessing:
    def __init__(self, train_data, test_data, target_size=(100, 100), batch_size=64):
        self.train_data = train_data
        self.test_data = test_data
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_datagen = ImageDataGenerator(
            rescale=1.0/255,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        self.validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
        self.labels = None

    def create_generators(self):
        train_generator = self.train_datagen.flow_from_directory(
            self.train_data,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        validation_generator = self.validation_datagen.flow_from_directory(
            self.test_data,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        self.labels = {value: key for key, value in train_generator.class_indices.items()}
        return train_generator, validation_generator

    def plot_sample_images(self, generator):
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
        idx = 0

        for i in range(2):
            for j in range(5):
                label = self.labels[np.argmax(generator[0][1][idx])]
                ax[i, j].set_title(f"{label}")
                ax[i, j].imshow(generator[0][0][idx])
                ax[i, j].axis("off")
                idx += 1

        plt.tight_layout()
        plt.suptitle("Sample Training Images", fontsize=21)
        plt.show()

class CNNModel:
    def __init__(self, input_shape=(100, 100, 3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=self.input_shape),
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

            Dense(units=self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

class ModelTrainer:
    def __init__(self, model, train_generator, validation_generator, epochs=20):
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.epochs = epochs
        self.callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]

    def train(self):
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            callbacks=self.callbacks
        )
        return history

    def plot_training_history(self, history):
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

    def evaluate(self, test_generator):
        return self.model.evaluate(test_generator)

    def predict(self, test_generator):
        return self.model.predict(test_generator)

    def plot_sample_predictions(self, test_generator, predictions, class_labels, num_samples=5):
        sample_indices = np.random.choice(range(len(predictions)), num_samples, replace=False)

        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(sample_indices):
            img_path = test_generator.filepaths[idx]
            img = plt.imread(img_path)

            true_label = class_labels[test_generator.classes[idx]]
            predicted_label = class_labels[np.argmax(predictions[idx])]

            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"True: {true_label}\nPred: {predicted_label}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example usage:
data_preprocessing = DataPreprocessing(train_data="D:/Virtual/Deep Learning/Multiclass classification cnn/seg_train", 
                                       test_data="D:/Virtual/Deep Learning/Multiclass classification cnn/seg_test")
train_generator, validation_generator = data_preprocessing.create_generators()
data_preprocessing.plot_sample_images(train_generator)

cnn_model = CNNModel(input_shape=(100, 100, 3), num_classes=6)
cnn_model.summary()

model_trainer = ModelTrainer(cnn_model.model, train_generator, validation_generator, epochs=20)
history = model_trainer.train()
model_trainer.plot_training_history(history)

# For evaluation and prediction
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    "D:/Virtual/Deep Learning/Multiclass classification cnn/seg_test",
    target_size=(100, 100),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

evaluation = model_trainer.evaluate(test_generator)
predictions = model_trainer.predict(test_generator)
class_labels = list(test_generator.class_indices.keys())

model_trainer.plot_sample_predictions(test_generator, predictions, class_labels, num_samples=5)
