import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from DataSet.model_zoo import ModelZoo

class ModelTrainer:
    def __init__(self, train_data_path, test_data_path, image_shape):
        self.train_data_path = train_data_path
        self.test_data_path  = test_data_path
        self.image_shape     = image_shape
        self.CLASS_NAMES     = ["F1", "F2", "F3"]

        self.model_zoo = ModelZoo(self.image_shape)

    def run(self, model_name):
        self.model_name = model_name
        self.generate_datasets()
        self.train_model()
        self.show_test_result()

    def generate_datasets(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            rotation_range=20
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_data_path,
            target_size=self.image_shape,
            batch_size=100,
            shuffle=False,
            class_mode="categorical",
            subset='training'
        )

        self.validation_generator = self.train_datagen.flow_from_directory(
            self.train_data_path,
            target_size=self.image_shape,
            batch_size=100,
            class_mode="categorical",
            classes=self.CLASS_NAMES,
            subset='validation'
        )

        self.test_generator = ImageDataGenerator(rescale=1. / 255)
        self.test_data = self.test_generator.flow_from_directory(
            self.test_data_path,
            target_size=self.image_shape,
            shuffle=False,
            batch_size=100,
            class_mode='categorical',
            classes=self.CLASS_NAMES
        )

    def train_model(self):
        self.model_zoo.select_model(self.model_name)
        self.model = self.model_zoo.model

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=30,
            epochs=300,
            validation_data=self.validation_generator,
            callbacks=self.model_zoo.my_callbakcs
        )

    def show_test_result(self):
        print(self.model.metrics_names)
        loss, accuracy = self.model.evaluate(self.test_data)
        print(f"Loss: {loss}\nAccuracy: {accuracy}")

if __name__ == "__main__":
    train_data_path = "/home/fitogether/Documents/fibrosis/DataSet/train_png"
    test_data_path  = "/home/fitogether/Documents/fibrosis/DataSet/test_png"
    image_shape     = (350, 350)

    model_trainer = ModelTrainer(train_data_path, test_data_path, image_shape)
    # model_trainer.run("alex_net")
    model_trainer.run("vgg_net")