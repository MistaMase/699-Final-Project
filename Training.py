#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import tensorflow as tf

import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

class FacialClassifier:

    def __init__(self, model_save_filename):
        self.training_filename = 'fer2013.csv'
        self.model_save_filename = model_save_filename
        self.model = None
        self._define_model()
        self.X = None
        self.Y = None
        self.N = None
        self.D = None
        self.num_class = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.learning_rate = 1e-3
        self.h = None

    # Load the input dataset
    def load_dataset(self):
        # Load the input dataset
        label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        names = ['emotion', 'pixels', 'usage']
        df = pd.read_csv(self.training_filename, names=names, na_filter=False)
        image_data = df['pixels']

        # images are 48x48
        # N = 35887
        self.Y = []     # Emotion
        self.X = []     # Pixel data
        first = True
        for line_idx, line in enumerate(open(self.training_filename)):
            if line_idx != 0:
                row = line.split(',')
                self.Y.append(int(row[0]))
                self.X.append([int(p) for p in row[1].split()])

        self.X, self.Y = np.array(self.X) / 255.0, np.array(self.Y)

        self.num_class = len(set(self.Y))
        print(self.num_class)

        self.N, self.D = self.X.shape
        self.X = self.X.reshape(self.N, 48, 48, 1)

    def train(self, destroy=False):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=0)
        self.y_train = (np.arange(self.num_class) == self.y_train[:, None]).astype(np.float32)
        self.y_test = (np.arange(self.num_class) == self.y_test[:, None]).astype(np.float32)

        if destroy:
            K.clear_session()

        K.set_value(self.model.optimizer.lr, self.learning_rate)

        self.h = self.model.fit(x=self.x_train,
                              y=self.y_train,
                              batch_size=64,
                              epochs=20,
                              verbose=1,
                              validation_data=(self.x_test, self.y_test),
                              shuffle=True,
                              callbacks=[ModelCheckpoint(filepath=self.model_save_filename)])

    def _define_model(self):
        self.model = Sequential()
        input_shape = (48, 48, 1)
        self.model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
        self.model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
        self.model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
        self.model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(7))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
        self.model.summary()
        print('Created Model')

if __name__ == '__main__':
    model_save_location = 'model_filter.h5'  # save model at this location after each epoch
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
    fc = FacialClassifier(model_save_location)
    print('Created Facial Classifier')
    #import tensorflow as tf
    #print(f'GPU Status: {tf.config.list_physical_devices("GPU")}')
    fc.load_dataset()
    print('Loaded Dataset')
    print('Starting Training')
    fc.train(True)
    print(f'Finished Training')

