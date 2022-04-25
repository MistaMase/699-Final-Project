#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers import BatchNormalization
import tensorflow.keras.backend as K
import tensorflow.keras.utils as util

from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize


class EmotionClassifier:

    def __init__(self, train_model=False, model_save_filename='model_filter.h5', logging_level=logging.INFO):
        self.logging_level = logging_level
        logging.basicConfig(level=logging_level, format='%(levelname)s: %(name)s:  %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.debug('Configured Logging')
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
        self.label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        if train_model:
            self.logger.info('Initialized Facial Classifier, No Model Present')
            self.load_dataset()
            self.train()

        else:
            self.logger.info('Initialized Facial Classifier: Using existing model %s', self.model_save_filename)
            self.model = keras.models.load_model(model_save_filename)

    # Load the input dataset
    def load_dataset(self):
        # Log that we're loading the dataset
        self.logger.info('Loading Dataset')

        # Load the input dataset
        column_names = ['emotion', 'pixels', 'usage']
        dataset = pd.read_csv(self.training_filename, names=column_names, na_filter=False)
        image_data = dataset['pixels']

        # images are 48x48
        # N = 35887
        self.Y = []     # Emotion
        self.X = []     # Pixel data
        for line_idx, line in enumerate(open(self.training_filename)):
            if line_idx != 0:
                row = line.split(',')
                self.Y.append(int(row[0]))
                self.X.append([int(p) for p in row[1].split()])

        # Convert to floating point values
        self.X, self.Y = np.array(self.X) / 255.0, np.array(self.Y)

        self.num_class = len(set(self.Y))
        self.logger.debug('Number of Classes: %s', self.num_class)

        self.N, self.D = self.X.shape
        self.X = self.X.reshape(self.N, 48, 48, 1)

    def train(self):
        self.logger.info('Training Model')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=0)
        self.y_train = (np.arange(self.num_class) == self.y_train[:, None]).astype(np.float32)
        self.y_test = (np.arange(self.num_class) == self.y_test[:, None]).astype(np.float32)

        K.set_value(self.model.optimizer.lr, self.learning_rate)

        self.h = self.model.fit(x=self.x_train,
                              y=self.y_train,
                              batch_size=64,
                              epochs=20,
                              verbose=1,
                              validation_data=(self.x_test, self.y_test),
                              shuffle=True,
                              callbacks=[ModelCheckpoint(filepath=self.model_save_filename)])

        self.logger.info('Finished Training Model')

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
        self.logger.debug('Created Model')

        # Stupid function automatically prints
        if (self.logging_level == logging.DEBUG):
            self.logger.debug(self.model.summary())

    def get_emotion(self, img):
        self.logger.debug('Running Classifier Prediction')
        # Resize image to 48, 48
        image_resized = resize(img, (48, 48), anti_aliasing=True)
        image_array = util.img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0
        result = self.model.predict(image_array)
        self.logger.info(result)
        labeled_result = {self.label_map[i]: result[0][i] for i in range(len(self.label_map))}
        return labeled_result
