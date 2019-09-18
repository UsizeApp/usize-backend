from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import glob
import pickle
import sys
import time
import datetime

nn_image_width = 160
nn_image_height = 160
split = 1800

#Chose images to train model
NORMAL = 1
NOISE = 1
MIRRORED = 0
MIRRORED_NOISE = 0

train_images = []
test_images = []
train_keypoints = []
test_keypoints = []

if NORMAL:
    with open('datapickles/front_images_arms_and_lower.pkl', 'rb') as file:
        normal_images, normal_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(normal_images)
            train_keypoints.append(normal_keypoints)
        else:
            train_images.append(normal_images[:split])
            train_keypoints.append(normal_keypoints[:split])
            test_images.append(normal_images[split:])
            test_keypoints.append(normal_keypoints[split:])

if NOISE:
    with open('datapickles/noise_front_images_arms_and_lower.pkl', 'rb') as file:
        noise_images, noise_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(noise_images)
            train_keypoints.append(noise_keypoints)
        else:
            train_images.append(noise_images[:split])
            train_keypoints.append(noise_keypoints[:split])
            test_images.append(noise_images[split:])
            test_keypoints.append(noise_keypoints[split:])

if MIRRORED:
    with open('datapickles/mirrored_front_images_arms_and_lower.pkl', 'rb') as file:
        mirrored_images, mirrored_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(mirrored_images)
            train_keypoints.append(mirrored_keypoints)
        else:
            train_images.append(mirrored_images[:split])
            train_keypoints.append(mirrored_keypoints[:split])
            test_images.append(mirrored_images[split:])
            test_keypoints.append(mirrored_keypoints[split:])

if MIRRORED_NOISE:
    with open('datapickles/mirrored_noise_front_images_arms_and_lower.pkl', 'rb') as file:
        mirrored_noise_images, mirrored_noise_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(mirrored_images)
            train_keypoints.append(mirrored_keypoints)
        else:
            train_images.append(mirrored_noise_images[:split])
            train_keypoints.append(mirrored_noise_keypoints[:split])
            test_images.append(mirrored_noise_images[split:])
            test_keypoints.append(mirrored_noise_keypoints[split:])

if not train_images:
    sys.exit()

#Creacion de la red neuronal convolucional:

model = keras.Sequential()

model.add(keras.layers.Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(nn_image_width, nn_image_height,1)))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(keras.layers.Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(keras.layers.LeakyReLU(alpha = 0.1))
model.add(keras.layers.BatchNormalization())


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(len(train_keypoints[0][0])))

model.summary()

model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae','acc'])

#sys.exit()
print(len(train_images))
print(len(train_keypoints))
print(len(test_images))
print(len(test_keypoints)) 

train_images = np.asarray(train_images)
train_images = train_images.reshape((-1, nn_image_width, nn_image_height,1))
print(train_images.shape)

train_keypoints = np.asarray(train_keypoints)
n_dataset, n_images, n_points = train_keypoints.shape
train_keypoints = train_keypoints.reshape((n_dataset*n_images,n_points))
print(train_keypoints.shape)

if split != -1:

    test_images = np.asarray(test_images)
    test_images = test_images.reshape((-1, nn_image_width, nn_image_height,1))
    print(test_images.shape)

    test_keypoints = np.asarray(test_keypoints)
    n_dataset, n_images, n_points = test_keypoints.shape
    test_keypoints = test_keypoints.reshape((n_dataset*n_images,n_points))
    print(test_keypoints.shape)

epochs = 2000
t_fit_inicial = time.time()
model.fit(train_images,train_keypoints, epochs = epochs, validation_data=None if split == -1 else (test_images, test_keypoints))
t_fit_final = time.time()
segundos = t_fit_final - t_fit_inicial
print("\nModel fit duró: {}\n".format(datetime.timedelta(seconds=segundos)))

#t_test_inicial = time.time()
#test_loss, test_acc = model.evaluate(test_images, test_keypoints)
#t_test_final = time.time()
#
#print('Test accuracy:', test_acc)
#print("\nTest duró: ", strftime("%H:%M:%S", t_test_final - t_test_inicial))
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model.save('models/UsizeNetConvolutional_front_{}-epochs_{}.h5'.format(epochs, timestamp))



