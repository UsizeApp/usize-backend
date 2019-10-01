from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

#Genera pickles de las imagenes y sus keypoints, a cada imagen se le aplica un resize y normalización para ser el input de la red neuronal.

def process_image(dataframe_row, images_folder_path, effect, plot = False):
    person_id = int(dataframe_row[0])
    points = dataframe_row[1:]
    if effect == "normal":
        image_path = "{}/{}.png".format(images_folder_path, person_id)
    elif effect == "mirrored":
        image_path = "{}/augmented_image_mirrored_noise_{}.png".format(images_folder_path, person_id)
    elif effect == "rotated":
        image_path = "{}/augmented_image_rotated_{}.png".format(images_folder_path, person_id)
    if not os.path.isfile(image_path):
        return [False, False, False]

    im = Image.open(image_path).convert('RGB') #RGB
    frame = cv2.imread(image_path)
    edges_channel = cv2.Canny(frame, int(max(0, np.mean(frame))), int(min(255, np.mean(frame))))
    #im = Image.open(images_folder_path + str(int(person_id)) + '.png').convert('L') #Grayscale 
    width, height = im.size
    
    if width < nn_image_width or height < nn_image_height:
        print("Error: id {} tiene muy baja resolucion, menor a {}x{} pixeles".format(person_id, nn_image_width, nn_image_height))
        return [False, False, False]
    
    im = np.asarray(im)
    
    #Add edges channel:
    edges_channel = np.reshape(edges_channel, (height, width, 1))
    im = np.concatenate((im, edges_channel), axis = 2)

    #Reshape and normalize image values
    im = cv2.resize(im, (nn_image_width, nn_image_height))
    im = im/255
    
    #Scale keypoints:
    points[::2] = np.round(points[::2] * (nn_image_width/width)) #scale x
    points[1::2] = np.round(points[1::2] * (nn_image_height/height)) #scale y
    if any(point > nn_image_width for point in points):
        print("Error: id {} tiene puntos fuera de la imagen.".format(person_id))
        return [False, False, False]

    if plot:
        # Para ver lo que se esta haciendo:
        plt.imshow(im, cmap='gray')
        plt.scatter(points[::2], points[1::2], s=20, marker='.', c='m')
        plt.show()
        exit()

    return [True, im, points]






nn_image_width = 160
nn_image_height = 160

#Variables para ver que pickles generar:
NORMAL = 1
MIRRORED_NOISE = 1
ROTATED = 1

#Que tipo de modelo generar:
MODEL = "all" #front o side o all

NUM_FILES_DESIRED = -1

front_images_folder_path = 'Usize_front_images_dataset/Images/'
side_images_folder_path = 'Usize_side_images_dataset/Images/'

if NORMAL:
    if MODEL == "front" or MODEL == "all":
        normal_front_keypoints_frame = pd.read_csv('Usize_front_images_dataset/normal_frontal_images_20_keypoints.csv')            
    if MODEL == "side" or MODEL == "all":
        normal_side_keypoints_frame = pd.read_csv('Usize_side_images_dataset/normal_side_images_9_keypoints.csv') 
if MIRRORED_NOISE:
    if MODEL == "front" or MODEL == "all":
        pass
        #reflected_front_keypoints_frame = pd.read_csv('Usize_front_images_dataset/mirrored_frontal_images_20_keypoints.csv')
    if MODEL == "side" or MODEL == "all":
        reflected_side_keypoints_frame = pd.read_csv('Usize_side_images_dataset/mirrored_side_images_9_keypoints.csv')
if ROTATED:
    if MODEL == "front" or MODEL == "all":
        rotated_front_keypoints_frame = pd.read_csv('Usize_front_images_dataset/rotated_frontal_images_20_keypoints.csv')
    if MODEL == "side" or MODEL == "all":
        rotated_side_keypoints_frame = pd.read_csv('Usize_side_images_dataset/rotated_side_images_9_keypoints.csv')
  



if NORMAL:
    print("Generando datapickle para imágenes normales")
    if MODEL == "front" or MODEL == "all":
        print("Imágenes frontales")
        images = []
        keypoints = []
        images_folder_path = os.path.join(front_images_folder_path, "normal")
        image_counter = 0
        for row in normal_front_keypoints_frame.values:
            status, image, points = process_image(row, images_folder_path, "normal")
            if status == False:
                continue
            images.append(image)
            keypoints.append(points)
            image_counter += 1
            if image_counter == NUM_FILES_DESIRED:
                break
        print("Normal front images: {}".format(len(images)))
        with open('datapickles/normal_front_images_RGB.pkl','wb') as file:
            pickle.dump([images, keypoints], file)
        print("Normal front data pickle created\n")

    if MODEL == "side" or MODEL == "all":
        print("Imágenes laterales")
        images = []
        keypoints = []
        images_folder_path = os.path.join(side_images_folder_path, "normal")
        image_counter = 0
        for row in normal_side_keypoints_frame.values:
            status, image, points = process_image(row, images_folder_path, "normal")
            if status == False:
                continue
            images.append(image)
            keypoints.append(points)
            image_counter += 1
            if image_counter == NUM_FILES_DESIRED:
                break
        print("Normal side images: {}".format(len(images)))
        with open('datapickles/normal_side_images_RGB.pkl','wb') as file:
            pickle.dump([images, keypoints], file)
        print("Normal side data pickle created\n")


if MIRRORED_NOISE:
    print("Generando datapickle para imagenes reflejadas con ruido")
    #if MODEL == "front" or MODEL == "all":
    #    print("Imágenes frontales")
    #    images = []
    #    keypoints = []
    #    images_folder_path = os.path.join(front_images_folder_path, "mirrored_noise")
    #    image_counter = 0
    #    for row in reflected_front_keypoints_frame.values:
    #        status, image, points = process_image(row, images_folder_path, "mirrored")
    #        if status == False:
    #            continue
    #        images.append(image)
    #        keypoints.append(points)
    #        image_counter += 1
    #        if image_counter == NUM_FILES_DESIRED:
    #            break
    #    print("Mirrored-noised front images: {}".format(len(images)))
    #    with open('datapickles/mirrored_noised_front_images_RGB.pkl','wb') as file:
    #        pickle.dump([images, keypoints], file)
    #    print("Mirrored-noised front data pickle created\n")
    if MODEL == "side" or MODEL == "all":
        print("Imágenes laterales")
        images = []
        keypoints = []
        images_folder_path = os.path.join(side_images_folder_path, "mirrored_noise")
        image_counter = 0
        for row in reflected_side_keypoints_frame.values:
            status, image, points = process_image(row, images_folder_path, "mirrored")
            if status == False:
                continue
            images.append(image)
            keypoints.append(points)
            image_counter += 1
            if image_counter == NUM_FILES_DESIRED:
                break
        print("Mirrored-noised side images: {}".format(len(images)))
        with open('datapickles/mirrored_noised_side_images_RGB.pkl','wb') as file:
            pickle.dump([images, keypoints], file)
        print("Mirrored-noised side data pickle created\n")

if ROTATED:
    print("Generando datapickle para imagenes rotadas")
    if MODEL == "front" or MODEL == "all":
        print("Imágenes frontales")
        images = []
        keypoints = []
        images_folder_path = os.path.join(front_images_folder_path, "normal")
        image_counter = 0
        for row in rotated_front_keypoints_frame.values:
            status, image, points = process_image(row, images_folder_path, "normal")
            if status == False:
                continue
            images.append(image)
            keypoints.append(points)
            image_counter += 1
            if image_counter == NUM_FILES_DESIRED:
                break
        print("Rotated front images: {}".format(len(images)))
        with open('datapickles/rotated_front_images_RGB.pkl','wb') as file:
            pickle.dump([images, keypoints], file)
        print("Rotated front data pickle created\n")
    if MODEL == "side" or MODEL == "all":
        print("Imágenes laterales")
        images = []
        keypoints = []
        images_folder_path = os.path.join(side_images_folder_path, "normal")
        image_counter = 0
        for row in rotated_side_keypoints_frame.values:
            status, image, points = process_image(row, images_folder_path, "normal")
            if status == False:
                continue
            images.append(image)
            keypoints.append(points)
            image_counter += 1
            if image_counter == NUM_FILES_DESIRED:
                break
        print("Rotated side images: {}".format(len(images)))
        with open('datapickles/rotated_side_images_RGB.pkl','wb') as file:
            pickle.dump([images, keypoints], file)
        print("Rotated side data pickle created\n")

