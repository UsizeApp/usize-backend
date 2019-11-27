from __future__ import absolute_import, division, print_function, unicode_literals

# Helper libraries
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import glob
from scipy.spatial import distance as dist
import time
import os
import random
import skimage as sk
import csv

def reflect_points(x_axis, points):
    #reflected_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    y_points = points[1::2]
    new_x_points = points[::2]
    for i in range(len(new_x_points)):
        diff = new_x_points[i] - x_axis
        new_x_points[i] -= 2*diff
    final_points = zip(new_x_points,y_points)
    return np.asarray(list(final_points)).flatten()

def rotate_points(image, degrees, points):
    angle = (degrees) * (np.pi/180)
    #np_image = np.asarray(image)
    #random_degree = random.uniform(-25, 25)
    #rotated_image = sk.transform.rotate(np_image, degrees)
    width, height = image.size
    rotated_image = image.rotate(degrees)
    center=(width / 2 - 0.5, height / 2 - 0.5)
    new_x_points = points[1::2] - center[0]
    new_y_points = points[::2] - center[1]
    new_x_points = np.cos(angle) * (new_x_points) - np.sin(angle) * (new_y_points) #x cos(x) - y sen(y)
    #new_x_points = np.cos(angle)*(new_x_points - center[0]) 
    new_y_points = np.sin(angle) * (new_x_points) + np.cos(angle) * (new_y_points) #x sen(x) + y cos(y)
    #new_y_points = np.sin(angle)*(new_y_points - center[1])
    new_x_points = new_x_points + center[0]
    new_y_points = new_y_points + center[1]
    final_points = zip(new_x_points,new_y_points)
    return [rotated_image, list(final_points)]

'''key_pts_frame = pd.read_csv('Usize_front_images_dataset/frontal_images_arms_and_lower_keypoints.csv')
data = key_pts_frame.values[0]
person_id = data[0]
keypoints = data[1:]
keypoints_cpy = np.copy(data[1:])
keypoints_cpy2 = np.copy(data[1:])
image = Image.open("Usize_front_images_dataset/Images/" + str(int(data[0])) + ".png")
width, height = image.size
center=(width / 2 - 0.5, height / 2 - 0.5)
reflected_image, reflected_keypoints = reflect_points(image, float(width/2), keypoints_cpy)
rotated_image, rotated_keypoints = rotate_points(image, 45, keypoints_cpy2)
reflected_keypoints = np.asarray(reflected_keypoints).flatten()
rotated_keypoints = np.asarray(rotated_keypoints).flatten()
plt.figure(1, figsize=(20,10))
plt.subplot(311)
plt.imshow(image)
plt.scatter(center[0], center[1], s=20, marker='.', c='orange')
plt.scatter(keypoints[1::2], keypoints[::2], s=20, marker='.', c='m')
plt.subplot(312)
plt.imshow(reflected_image)
plt.scatter(reflected_keypoints[::2], reflected_keypoints[1::2], s=20, marker='.', c='m')
plt.subplot(313)
plt.imshow(rotated_image)
plt.scatter(center[0], center[1], s=20, marker='.', c='orange')
plt.scatter(rotated_keypoints[::2], rotated_keypoints[1::2], s=20, marker='.', c='m')
plt.show()'''

def reflect_keypoints_csv(path, side = 0):
    csv_path = path + '.csv'
    key_pts_frame = pd.read_csv(csv_path)
    with open(path + 'REFLECTED' + '.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',')
        if side:
            row = ['id','x1','y1','x2','y2', 'x3','y3','x4','y4','x5','y5','x6','y6','x7','y7','x8','y8','x9','y9','x10','y10','x11','y11']
        else:
            row = ['id','x1','y1','x2','y2', 'x3','y3','x4','y4','x5','y5','x6','y6','x7','y7','x8','y8','x9','y9','x10','y10', 'x11','y11','x12','y12','x13','y13','x14','y14','x15','y15','x16','y16','x17','y17','x18','y18','x19','y19','x20','y20','x21','y21','x22','y22']
        file_writer.writerow(row)
        for data in key_pts_frame.values:
            person_id = data[0]
            if person_id == 19580:
                continue
            points = data[1:]
            keypoints = np.copy(points)
            if side:
                im = Image.open("extra_dataset/side/normal/" + str(int(person_id)) + "_side.jpg")
            else:
                #ACTUALIZAR ANTES DE CORRER PARA EL EXTRA DATASET FRONTAL
                im = Image.open("Usize_front_images_dataset/Images/" + str(int(person_id)) + ".png")
            width, height = im.size
            data[1:] = reflect_points(float(width/2), keypoints)
            file_writer.writerow(data)
    return

#reflect_keypoints_csv('Usize_side_images_dataset/side_images_9_keypoints', 1)
reflect_keypoints_csv('extra_dataset/side_images_extra_dataset_11_keypoints', 1)
