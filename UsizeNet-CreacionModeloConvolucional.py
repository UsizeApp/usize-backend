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
from sklearn.model_selection import train_test_split

#Parameters
nn_image_width = 160
nn_image_height = 160
split = 0.8
epochs = 500
show = 1

#Chose images to train model
NORMAL = 0
NOISE = 0
MIRRORED = 0 #Use only on side model
MIRRORED_NOISE = 0 #Use only on side model
ROTATED = 0
EXTRA = 1
EXTRA_MIRRORED_NOISE = 1 #Use only on side model

#Chose model to create:
MODEL = "side" #front o side

train_images = []
test_images = []
train_keypoints = []
test_keypoints = []

if NORMAL:
    with open('datapickles/'+ MODEL +'_normal_images_RGB.pkl', 'rb') as file:
        normal_images, normal_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(normal_images)
            train_keypoints.append(normal_keypoints)
        else:
            normal_images_train, normal_images_test, normal_keypoints_train, normal_keypoints_test = train_test_split(normal_images, normal_keypoints, train_size=split, random_state=42) 
            train_images.append(normal_images_train)
            train_keypoints.append(normal_keypoints_train)
            test_images.append(normal_images_test)
            test_keypoints.append(normal_keypoints_test)


if EXTRA:
    with open('datapickles/'+ MODEL +'_extra_images.pkl', 'rb') as file:
        normal_images, normal_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(normal_images)
            train_keypoints.append(normal_keypoints)
        else:
            normal_images_train, normal_images_test, normal_keypoints_train, normal_keypoints_test = train_test_split(normal_images, normal_keypoints, train_size=split, random_state=42) 
            train_images.append(normal_images_train)
            train_keypoints.append(normal_keypoints_train)
            test_images.append(normal_images_test)
            test_keypoints.append(normal_keypoints_test)

if EXTRA_MIRRORED_NOISE:
    with open('datapickles/'+ MODEL +'_extra_mirrored_noise_images.pkl', 'rb') as file:
        mirrored_noise_images, mirrored_noise_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(mirrored_noise_images)
            train_keypoints.append(mirrored_noise_keypoints)
        else:
            mirrored_noise_images_train, mirrored_noise_images_test, mirrored_noise_keypoints_train, mirrored_noise_keypoints_test = train_test_split(mirrored_noise_images, mirrored_noise_keypoints, train_size=split, random_state=42) 
            train_images.append(mirrored_noise_images_train)
            train_keypoints.append(mirrored_noise_keypoints_train)
            test_images.append(mirrored_noise_images_test)
            test_keypoints.append(mirrored_noise_keypoints_test)


if NOISE:
    with open('datapickles/'+ MODEL +'_noise_images.pkl', 'rb') as file:
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
    with open('datapickles/'+ MODEL +'_mirrored_images.pkl', 'rb') as file:
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
    with open('datapickles/'+ MODEL +'_mirrored_noise_images.pkl', 'rb') as file:
        mirrored_noise_images, mirrored_noise_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(mirrored_noise_images)
            train_keypoints.append(mirrored_noise_keypoints)
        else:
            mirrored_noise_images_train, mirrored_noise_images_test, mirrored_noise_keypoints_train, mirrored_noise_keypoints_test = train_test_split(mirrored_noise_images, mirrored_noise_keypoints, train_size=split, random_state=42) 
            train_images.append(mirrored_noise_images_train)
            train_keypoints.append(mirrored_noise_keypoints_train)
            test_images.append(mirrored_noise_images_test)
            test_keypoints.append(mirrored_noise_keypoints_test)

if ROTATED:
    with open('datapickles/'+ MODEL +'_rotated_images.pkl', 'rb') as file:
        rotated_images, rotated_keypoints = pickle.load(file)
        if split == -1:
            train_images.append(rotated_images)
            train_keypoints.append(rotated_keypoints)
        else:
            rotated_images_train, rotated_images_test, rotated_keypoints_train, rotated_keypoints_test = train_test_split(rotated_images, rotated_keypoints, train_size=split, random_state=42) 
            train_images.append(rotated_images_train)
            train_keypoints.append(rotated_keypoints_train)
            test_images.append(rotated_images_test)
            test_keypoints.append(rotated_keypoints_test)

if not train_images:
    sys.exit()

width, height, channels = train_images[0][0].shape

#Creacion de la red neuronal convolucional:

model = keras.Sequential()

model.add(keras.layers.Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(nn_image_width, nn_image_height,channels)))
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

if len(train_images) > 1:
    temp = np.array(train_images[0])
    for images in train_images[1:]:
        temp = np.concatenate((temp,np.array(images)),axis = 0)
    train_images = temp
else:
    train_images = np.array(train_images)
train_images = train_images.reshape((-1, nn_image_width, nn_image_height,channels))
print(train_images.shape)

train_keypoints = np.asarray(train_keypoints)
if len(train_keypoints) > 1:
    temp = np.array(train_keypoints[0])
    for images in train_keypoints[1:]:
        temp = np.concatenate((temp,np.array(images)),axis = 0)
    train_keypoints = temp
else:
    train_keypoints = np.array(train_keypoints)
    n_dataset, n_images, n_points = train_keypoints.shape
    train_keypoints = train_keypoints.reshape((n_dataset*n_images,n_points))
print("aca")
print(train_keypoints.shape)
print("sali")

if split != -1:
    print("split")
    
    if len(test_images) > 1:
        temp = np.array(test_images[0])
        for images in test_images[1:]:
            print("primer for")
            temp = np.concatenate((temp,np.array(images)),axis = 0)
        test_images = temp
    else:
        test_images = np.array(test_images[0])
    print(test_images.shape)

    if len(test_keypoints) > 1:
        temp = np.array(test_keypoints[0])
        for images in test_keypoints[1:]:
            print("segundo for")
            temp = np.concatenate((temp,np.array(images)),axis = 0)
        test_keypoints = temp
    else:
        test_keypoints = np.array(test_keypoints)
        n_dataset, n_images, n_points = test_keypoints.shape
        test_keypoints = test_keypoints.reshape((n_dataset*n_images,n_points))
    print(test_keypoints.shape)

#epochs = 2000
t_fit_inicial = time.time()
history = model.fit(train_images,train_keypoints, epochs = epochs, validation_data=None if split == -1 else (test_images, test_keypoints))
t_fit_final = time.time()
segundos = t_fit_final - t_fit_inicial
print("\nModel fit duró: {}\n".format(datetime.timedelta(seconds = segundos)))

#t_test_inicial = time.time()
#test_loss, test_acc = model.evaluate(test_images,test_keypoints)
#t_test_final = time.time()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")

model.save('models/UsizeNetConvolutional_{}_{}-channels_{}-epochs_{}.h5'.format(MODEL,channels,epochs,timestamp))
print(MODEL + " model created")

if show:
    n_epochs = epochs
    acc = history.history['loss']
    val_acc = history.history['val_loss']
    plt.plot(np.arange(1,n_epochs + 1),acc, label = 'loss')
    plt.plot(np.arange(1,n_epochs + 1),val_acc, label = 'val_loss')
    plt.legend()
    plt.savefig("latest {} model history.png".format(MODEL))
    plt.show()

#test_loss,test_mae_loss,test_acc = model.evaluate(test_images, test_keypoints)

#print('Test accuracy:', test_acc)
#print(\nTest duró: ", strftime.strftime("%H:%M:%S", t_test_final - t_test_inicial))



