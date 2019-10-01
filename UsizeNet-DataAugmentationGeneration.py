from os import listdir
from pathlib import Path
from PIL import Image
from scipy import ndarray
from skimage import io
from skimage import transform
from skimage import util
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import skimage as sk

def rotate_points(points, width, height, degrees):
    """Use numpy to build a rotation matrix and take the dot product."""

    # puntos originales
    X_orig = points[0]
    Y_orig = points[1]

    # trasladar puntos con respecto a las dimensiones de la imagen
    X_translated = list(map(lambda x: x - width/2,  X_orig))
    Y_translated = list(map(lambda y: y - height/2, Y_orig))

    # a radianes
    radians = math.radians(degrees)

    # matrices de rotacion
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])

    X_rotated = []
    Y_rotated = []

    for i in range(len(X_orig)):
        x = X_translated[i]
        y = Y_translated[i]
        m = np.dot(j, [x, y])
        x_rot_val = float(m.T[0])
        y_rot_val = float(m.T[1])
        X_rotated.append(x_rot_val)
        Y_rotated.append(y_rot_val)

    # trasladar puntos (de vuelta) con respecto a las dimensiones de la imagen
    X_final = list(map(lambda x: x + width/2,  X_rotated))
    Y_final = list(map(lambda y: y + height/2, Y_rotated))
    rotated_points = [X_final, Y_final]
    return rotated_points


def rotate_image(image_array: ndarray, degree):
    return sk.transform.rotate(image_array, degree)


def draw_points(image_file, points, degrees = 0, rotate = False):
    image = Image.open(image_file)
    if rotate == True:
        image = image.rotate(degrees)
        plt.imshow(image)
        plt.scatter(points[0], points[1], c='#73ff00', s=1)
        plt.savefig('test_rotation.png', dpi=300)
    else:
        plt.imshow(image)
        plt.scatter(points[0], points[1], c='orange', s=1)
        plt.savefig('test.png', dpi=300)
    plt.clf()


def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def main(num_files_desired):
    if MODEL == "front":
        folder_path = 'Usize_front_images_dataset/Images'
        keypoints_df = pd.read_csv('Usize_front_images_dataset/frontal_images_20_keypoints.csv')
    elif MODEL == "side":
        folder_path = 'Usize_side_images_dataset/Images'
        keypoints_df = pd.read_csv('Usize_side_images_dataset/side_images_9_keypoints.csv')

    # find all files paths from the folder
    normal_path = os.path.join(folder_path, "normal")
    images = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, f))]
    total = len(images)
    current = 1
    new_points = []
    for image_path in images:
        if num_files_desired == 0: break
        print("{}/{}".format(current, total))

        # image_id
        image_id = int(Path(image_path).stem)
        #image_id = int(image_path.replace(".png","").replace("Usize_side_images_dataset/side_images/",""))

        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)

        if EFFECT == "mirror" or EFFECT == "all":
            mirrored = horizontal_flip(image_to_transform)
            mirrored_noise = random_noise(mirrored)
            mirrored_noise *= 255
            final_image = mirrored_noise.astype(np.uint8)
            # write transformations to disk
            try:
                # guardar imagen
                new_file_path = '{}/mirrored_noise/augmented_image_mirrored_noise_{}.png'.format(folder_path, image_id)
                io.imsave(new_file_path, final_image)
            except OSError as err:
                print("Error con la imagen {}, {}".format(image_id, err))

        if EFFECT == "rotation" or EFFECT == "all":
            original_points = keypoints_df.loc[keypoints_df['id'] == image_id].values.tolist()[0]
            X_orig = original_points[1::2]
            Y_orig = original_points[2::2]

            # pick a random degree of rotation between -25 and 25
            random_degree = random.randrange(-25, 25)

            # image dimensions
            height, width, _ = image_to_transform.shape

            rotated_points = rotate_points([X_orig, Y_orig], width, height, random_degree)
            rotated_image  = rotate_image(image_to_transform, random_degree)
            # Para que me deje de dar el error de:
            # "Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning."
            rotated_image *= 255
            final_image = rotated_image.astype(np.uint8)

            # write transformations to disk
            try:
                # guardar imagen
                new_file_path = '{}/rotated/augmented_image_rotated_{}.png'.format(folder_path, image_id)
                io.imsave(new_file_path, final_image)

                # guardar nuevos puntos
                points = [image_id]
                X_rotated = rotated_points[0]
                Y_rotated = rotated_points[1]
                for i in range(len(X_rotated)):
                    points.append(X_rotated[i])
                    points.append(Y_rotated[i])
                new_points.append(points)

            except OSError as err:
                print("Error con la imagen {}, {}".format(image_id, err))
        
        num_files_desired -= 1
        current += 1

    if EFFECT == "rotation" or EFFECT == "all":
        if MODEL == "front":
            columns = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20"]
        elif MODEL == "side":
            columns = ["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9"]
        
        new_df = pd.DataFrame(new_points, columns = columns)
        if MODEL == "front":
            new_df.to_csv('Usize_front_images_dataset/rotated_frontal_images_20_keypoints.csv', index=False)
        elif MODEL == "side":
            new_df.to_csv('Usize_side_images_dataset/rotated_side_images_9_keypoints.csv', index=False)






MODEL = "side" # front o side
EFFECT = "mirror" # mirror o rotation o all
num_files_desired = -1 # -1 para todas las imagenes en la carpeta
main(num_files_desired)