import os
import random
from scipy import ndarray

# reading points
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

rotated_points  = []
mirrored_points = []


def random_rotation(image_array: ndarray, points):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return [sk.transform.rotate(image_array, random_degree), sk.transform.rotate(points, random_degree)]

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def rotate_array(points, image_id):
    rotated = [image_id]
    # hacer cosas
    rotated_points.append(rotated)

def mirror_array(points, image_id):
    mirrored = [image_id]
    # hacer cosas
    mirrored_points.append(mirrored)

folder_path = 'Usize_front_images_dataset/Images'
num_files_desired = -1 # -1 para todas las imagenes en la carpeta

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

df = pd.read_csv('Usize_front_images_dataset/frontal_images_arms_and_lower_keypoints.csv')
total = len(images)
current = 1
for image_path in images:
    print("{}/{}".format(current,total))
    if num_files_desired == 0: break

    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)

    # image_id
    image_id = int(image_path.replace(".png","").replace("Usize_front_images_dataset/Images/",""))

    # row of points 
    points = df[df.id == image_id].drop('id', axis=1)
    x_axis = points.values[0][0::2]
    y_axis = points.values[0][1::2]

    points_matrix = np.column_stack((x_axis,y_axis))

    #alt = sk.transform.rotate(points_matrix, 180)

    # random transformation to apply for a single image
    rotated_image, rotated_points = random_rotation(image_to_transform, points_matrix)
    x_axis_rotated = rotated_points.flatten()[0::2]
    y_axis_rotated = rotated_points.flatten()[1::2]

    mirrored = horizontal_flip(image_to_transform)
    #mirror_array(points, image_id)

    noise = random_noise(image_to_transform)

    new_file_path_rotated   = '{}/output_augmented/augmented_image_rotated_{}.png'.format(folder_path, image_id)
    new_file_path_noise     = '{}/output_augmented/augmented_image_noise_{}.png'.format(folder_path, image_id)
    new_file_path_mirrored  = '{}/output_augmented/augmented_image_mirrored_{}.png'.format(folder_path, image_id)

    # write transformations to disk
    try:
        #io.imsave(new_file_path_rotated, rotated_image)
        #io.imsave(new_file_path_mirrored, mirrored)
        io.imsave(new_file_path_noise, noise)
        num_files_desired -= 1
    except OSError as err:
        print("Error con la imagen {}, {}".format(image_id, err))

    current += 1

    #original_image = Image.open(image_path)
    #plt.imshow(original_image)
    #plt.scatter(y_axis, x_axis, s=20, marker='.', c='m')
    #plt.show()

    #rotated_image = Image.open(new_file_path_rotated)
    #plt.imshow(rotated_image)
    #plt.scatter(x_axis_rotated, y_axis_rotated, s=20, marker='.', c='m')
    #plt.show()