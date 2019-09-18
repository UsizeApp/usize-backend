from __future__ import absolute_import, division, print_function, unicode_literals

# Helper libraries
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import glob

import pickle
import sys

#Genera pickles de las imagenes y sus keypoints, a cada imagen se le aplica un resize, grayscale y normalización para ser el input de la 
#red neuronal.

nn_image_width = 160
nn_image_height = 160

#Variables para ver que pickles generar:
NORMAL = 1
NOISE = 1
MIRRORED = 0 #Dejar en 0
MIRRORED_NOISE = 0 #Dejar en 0

if NORMAL or NOISE:
    normal_keypoints_frame = pd.read_csv('Usize_front_images_dataset/frontal_images_20_keypoints.csv')            
if MIRRORED or MIRRORED_NOISE:
    reflected_keypoints_frame = pd.read_csv('Usize_front_images_dataset/frontal_images_reflected_arms_and_lower_keypoints.csv')

i = 0
        
#Get normal images data
error_ids = []
if NORMAL:
    images = []
    keypoints = []
    for row in normal_keypoints_frame.values:
        person_id = row[0]
        if person_id in error_ids:
            continue
        points = row[1:]
        im = Image.open('Usize_front_images_dataset/Images/' + str(int(person_id)) + '.png').convert('L')
        w,h = im.size
        if w < nn_image_width or h < nn_image_height:
            error_ids.append(person_id)
            print("Error: id " + str(int(person_id)) + " tiene muy baja resolucion.")
            continue
        im = im.resize((nn_image_width,nn_image_height)) 
        im = np.asarray(im)
        im = im/255 #Normalize color values
        #scale keypoints:
        points[::2] = np.round(points[::2] * (nn_image_width/w)) #scale x
        points[1::2] = np.round(points[1::2] * (nn_image_height/h)) #scale y
        if any(point > nn_image_width for point in points):
            error_ids.append(person_id)
            print("Error: id: " + str(int(person_id)) + " tiene puntos fuera de la imagen.")
            continue
        #Para ver lo que se esta haciendo:
        #plt.imshow(im,cmap='gray')
        #plt.scatter(points[::2], points[1::2], s=20, marker='.', c='m')
        #plt.show()
        #break

        images.append(im)
        keypoints.append(points)
        i += 1
    print("Normal images:" + str(len(images)))
    with open('datapickles/front_images_arms_and_lower.pkl','wb') as file:
        pickle.dump([images,keypoints],file)
    print("Normal data pickle created")

if NOISE:
    images = []
    keypoints = []
    for row in normal_keypoints_frame.values:
        person_id = row[0]
        if person_id in error_ids:
            print("Skipping id " + str(int(person_id)))
            continue
        points = row[1:]
        im = Image.open('Usize_front_images_dataset/Images/output_augmented/augmented_image_noise_' + str(int(person_id)) + '.png').convert('L')
        w,h = im.size
        if w < nn_image_width or h < nn_image_height:
            error_ids.append(person_id)
            print("Error: id " + str(int(person_id)) + " tiene muy baja resolucion.")
            continue
        im = im.resize((nn_image_width,nn_image_height)) 
        im = np.asarray(im)
        im = im/255 #Normalize color values
        #scale keypoints:
        points[::2] = np.round(points[::2] * (nn_image_width/w)) #scale x
        points[1::2] = np.round(points[1::2] * (nn_image_height/h)) #scale y
        if any(point > nn_image_width for point in points):
            error_ids.append(person_id)
            print("Error: id: " + str(int(person_id)) + " tiene puntos fuera de la imagen.")
            continue            
        #Para ver lo que se esta haciendo:
        #plt.imshow(im,cmap='gray')
        #plt.scatter(points[::2], points[1::2], s=20, marker='.', c='m')
        #plt.show()
        #break
                    
        images.append(im)
        keypoints.append(points)
        i += 1
    print("Noise images:" + str(len(images)))
    with open('datapickles/noise_front_images_arms_and_lower.pkl','wb') as file:
        pickle.dump([images,keypoints],file)
    print("Noise data pickle created")

if MIRRORED:
    images = []
    keypoints = []
    for row in reflected_keypoints_frame.values:
        person_id = row[0]
        if person_id in error_ids:
            print("Skipping id " + str(int(person_id)))
            continue
        points = row[1:]
        im = Image.open('Usize_front_images_dataset/Images/output_augmented/augmented_image_mirrored_' + str(int(person_id)) + '.png').convert('L')
        w,h = im.size
        if w < nn_image_width or h < nn_image_height:
            error_ids.append(person_id)
            print("Error: id " + str(int(person_id)) + " tiene muy baja resolucion.")
            continue
        im = im.resize((nn_image_width,nn_image_height)) 
        im = np.asarray(im)
        im = im/255 #Normalize color values
        #scale keypoints:
        points[::2] = np.round(points[::2] * (nn_image_width/w)) #scale x
        points[1::2] = np.round(points[1::2] * (nn_image_height/h)) #scale y
        if any(point > nn_image_width for point in points):
            error_ids.append(person_id)
            print("Error: id: " + str(int(person_id)) + " tiene puntos fuera de la imagen.")
            continue            
        #Para ver lo que se esta haciendo:
        #plt.imshow(im,cmap='gray')
        #plt.scatter(points[::2], points[1::2], s=20, marker='.', c='m')
        #plt.show()
        #break    

        images.append(im)
        keypoints.append(points)
        i += 1
    
    print("Mirrored images:" + str(len(images)))
    with open('datapickles/mirrored_front_images_arms_and_lower.pkl','wb') as file:
        pickle.dump([images,keypoints],file)
    print("Mirrored data pickle created")

if MIRRORED_NOISE:
    images = []
    keypoints = []
    for row in reflected_keypoints_frame.values:
        person_id = row[0]
        if person_id in error_ids:
            print("Skipping id " + str(int(person_id)))
            continue
        points = row[1:]
        im = Image.open('Usize_front_images_dataset/Images/output_augmented/augmented_image_mirrored_noise' + str(int(person_id)) + '.png').convert('L')
        w,h = im.size
        if w < nn_image_width or h < nn_image_height:
            error_ids.append(person_id)
            print("Error: id " + str(int(person_id)) + " tiene muy baja resolucion.")
            continue
        im = im.resize((nn_image_width,nn_image_height)) 
        im = np.asarray(im)
        im = im/255 #Normalize color values
        #scale keypoints:
        points[::2] = np.round(points[::2] * (nn_image_width/w)) #scale x
        points[1::2] = np.round(points[1::2] * (nn_image_height/h)) #scale y
        if any(point > nn_image_width for point in points):
            error_ids.append(person_id)
            print(person_id)
            continue      
        #Para ver lo que se esta haciendo:
        #plt.imshow(im,cmap='gray')
        #plt.scatter(points[::2], points[1::2], s=20, marker='.', c='m')
        #plt.show()
        #break    

        images.append(im)
        keypoints.append(points)
        i += 1
    print("Noise mirrored images:" + str(len(images)))
    with open('datapickles/mirrored_noise_front_images_arms_and_lower.pkl','wb') as file:
        pickle.dump([images,keypoints],file)
    print("Mirrored data pickle created")