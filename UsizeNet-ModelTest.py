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
from scipy.spatial import distance as dist
import time
import os

MODE = "KEYPOINTS" #KEYPOINTS o HEATMAP

def plot_line(p1,p2,color = 'orange'):
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]], c = color)
    return

def midpoint(p1, p2):
    if (p1 == None or p2 == None):
        return -1
    return [int((p1[0]+p2[0])/2) , int((p1[1]+p2[1])/2)]

nn_width = 160
nn_height = 160

def get_front_measurements(file, body_height_cm, show = False):
    
    original_image = Image.open(file)
    width, height = original_image.size
    print(height,width)
    feeded_image = original_image.convert('L')
    feeded_image = feeded_image.resize((nn_width,nn_height))
    feeded_image = np.asarray(feeded_image)/255
    feeded_image = feeded_image.reshape(1,nn_width,nn_height,1) # if seq NN feeded_image.reshape(1,nn_width*nn_height)

    t = time.time()
    if MODE == "HEATMAP":
        model = keras.models.load_model('UsizeNetConvolutional_front_Heatmap.h5')
        predicted_heatmap = model.predict(feeded_image) 
        predicted_heatmap = predicted_heatmap[0].reshape(nn_width,nn_height)
        print(predicted_heatmap.shape)
        print(np.amax(predicted_heatmap))
        return

    if MODE == "KEYPOINTS":
        model = keras.models.load_model('models/UsizeNetConvolutional_front_2000-epochs_2019-09-15 21_54_52.h5')    
        predicted_keypoints = model.predict(feeded_image)
        predicted_keypoints = predicted_keypoints[0]

        #plt.imshow(original_image.convert('L').resize((nn_width,nn_height)))
        #plt.scatter(predicted_keypoints[::2], predicted_keypoints[1::2], s=20, marker='.', c='m')
        #plt.show()

        predicted_keypoints[::2] = np.round(predicted_keypoints[::2] * (width/nn_width))
        predicted_keypoints[1::2] = np.round(predicted_keypoints[1::2] * (height/nn_height))

        #What is each point:

        right_wrist = [predicted_keypoints[0],predicted_keypoints[1]]
        right_elbow = [predicted_keypoints[2],predicted_keypoints[3]]
        right_shoulder1 = [predicted_keypoints[4],predicted_keypoints[5]]
        right_shoulder2 = [predicted_keypoints[6],predicted_keypoints[7]]
        head_right = [predicted_keypoints[8],predicted_keypoints[9]]
        head_left = [predicted_keypoints[10],predicted_keypoints[11]]
        left_shoulder2 = [predicted_keypoints[12],predicted_keypoints[13]]
        left_shoulder1 = [predicted_keypoints[14],predicted_keypoints[15]]
        left_elbow = [predicted_keypoints[16],predicted_keypoints[17]]
        left_wrist = [predicted_keypoints[18],predicted_keypoints[19]]
        left_armpit = [predicted_keypoints[20],predicted_keypoints[21]]
        left_chest = [predicted_keypoints[22],predicted_keypoints[23]] 
        left_hip = [predicted_keypoints[24],predicted_keypoints[25]]
        left_knee = [predicted_keypoints[26],predicted_keypoints[27]]
        outer_left_ankle = [predicted_keypoints[28],predicted_keypoints[29]]
        outer_right_ankle = [predicted_keypoints[30],predicted_keypoints[31]]
        right_knee = [predicted_keypoints[32],predicted_keypoints[33]]
        right_hip = [predicted_keypoints[34],predicted_keypoints[35]]
        right_chest = [predicted_keypoints[36],predicted_keypoints[37]]
        right_armpit = [predicted_keypoints[38],predicted_keypoints[39]]

    #Extra points:

    floor = midpoint(outer_left_ankle,outer_right_ankle)
    head_upper = midpoint(head_right,head_left)

    #Height:
    height_in_pixels = dist.euclidean(floor, head_upper)
    pixelsPerMetric = height_in_pixels/body_height_cm

    #Lengths:

    #Arms
    right_humerus = dist.euclidean(right_shoulder1, right_elbow)
    right_forearm = dist.euclidean(right_elbow, right_wrist)
    left_humerus = dist.euclidean(left_shoulder1, left_elbow)
    left_forearm = dist.euclidean(left_elbow, left_wrist)
    right_arm = (right_humerus + right_forearm) / pixelsPerMetric
    left_arm = (left_humerus + left_forearm) / pixelsPerMetric

    #Legs
    right_femur = dist.euclidean(right_hip,right_knee)
    right_shin = dist.euclidean(right_knee, outer_right_ankle)
    left_femur = dist.euclidean(left_hip,left_knee)
    left_shin = dist.euclidean(left_knee, outer_left_ankle)
    right_leg = (right_femur + right_shin) / pixelsPerMetric
    left_leg = (left_femur + left_shin) / pixelsPerMetric

    #Hips
    hip_length = dist.euclidean(right_hip,left_hip) / pixelsPerMetric

    print("Brazo derecho: " + str(right_arm))
    print("Brazo izquierdo: " + str(left_arm))
    print("Pierna derecha: " + str(right_leg))
    print("Pierna izquierda: " + str(left_leg))
    print("Caderas: " + str(hip_length))

    if show:
        
        plt.imshow(original_image)
        plt.scatter(predicted_keypoints[::2], predicted_keypoints[1::2], s=20, marker='.', c='m')
        #plt.scatter(original_keypoints[1::2], original_keypoints[::2], s=20, marker='.', c='orange')

        #Height line
        plot_line(floor,head_upper)

        #Legs lines
        plot_line(right_hip,right_knee)
        plot_line(right_knee,outer_right_ankle)
        plot_line(left_hip,left_knee)
        plot_line(left_knee,outer_left_ankle)

        #Arms lines
        plot_line(right_shoulder1,right_elbow)
        plot_line(right_elbow,right_wrist)
        plot_line(left_shoulder1,left_elbow)
        plot_line(left_elbow,left_wrist)

        #Hip lines
        plot_line(right_hip,left_hip)

        plt.show()

    front_measures = {
        "right_arm": right_arm,
        "left_arm": left_arm,
        "right_leg": right_leg,
        "left_leg": left_leg,
        "hip_length": hip_length,
        "path": os.path.abspath('output/Output-Skeleton.jpg'),
        "time": "{:.3f}".format(time.time() - t)
    }
    return front_measures

def get_side_measurements(file, body_height_cm, show = False):
    original_image = Image.open(file)
    width, height = original_image.size
    print(height,width)
    feeded_image = original_image.convert('L')
    feeded_image = feeded_image.resize((nn_width,nn_height))
    feeded_image = np.asarray(feeded_image)/255
    feeded_image = feeded_image.reshape(1,nn_width,nn_height,1) # if seq NN feeded_image.reshape(1,nn_width*nn_height)

    t = time.time()

    if MODE == "KEYPOINTS":
        model = keras.models.load_model('models/UsizeNetConvolutional_side.h5')    
        predicted_keypoints = model.predict(feeded_image)
        predicted_keypoints = predicted_keypoints[0]

        #plt.imshow(original_image.convert('L').resize((nn_width,nn_height)))
        #plt.scatter(predicted_keypoints[::2], predicted_keypoints[1::2], s=20, marker='.', c='m')
        #plt.show()

        predicted_keypoints[::2] = np.round(predicted_keypoints[::2] * (width/nn_width))
        predicted_keypoints[1::2] = np.round(predicted_keypoints[1::2] * (height/nn_height))

        #What is each point:

        chest = [predicted_keypoints[0],predicted_keypoints[1]]
        upper_bust = [predicted_keypoints[2],predicted_keypoints[3]]
        lower_bust = [predicted_keypoints[4],predicted_keypoints[5]]
        front_hip = [predicted_keypoints[6],predicted_keypoints[7]]
        heel = [predicted_keypoints[8],predicted_keypoints[9]]
        back_hip = [predicted_keypoints[10],predicted_keypoints[11]]
        lower_back = [predicted_keypoints[12],predicted_keypoints[13]]
        upper_back = [predicted_keypoints[14],predicted_keypoints[15]]
        head = [predicted_keypoints[16],predicted_keypoints[17]]

    #Height:
    height_in_pixels = dist.euclidean(heel, head)
    pixelsPerMetric = height_in_pixels/body_height_cm

    #Lengths:

    #Side hip
    hip_side_length = dist.euclidean(front_hip, back_hip)/pixelsPerMetric
    
    #Side chest
    chest_depth = dist.euclidean(chest, upper_back)/pixelsPerMetric

    #Bust
    bust_depth = dist.euclidean(upper_bust, lower_back)/pixelsPerMetric

    print("Cadera lateral: " + str(hip_side_length))
    print("Pecho: " + str(chest_depth))
    print("Busto: " + str(bust_depth))
    
    if show:
        plt.imshow(original_image)
        plt.scatter(predicted_keypoints[::2], predicted_keypoints[1::2], s=20, marker='.', c='m')
        #plt.scatter(original_keypoints[1::2], original_keypoints[::2], s=20, marker='.', c='orange')

        #Height line
        plot_line(heel,head)

        #Hip line
        plot_line(front_hip, back_hip)

        #Chest line
        plot_line(chest, upper_back)

        #Bust line
        plot_line(upper_bust, lower_back)

        plt.show()
        
    side_measures = {
        "hip_side_length": hip_side_length,
        "chest_depth": chest_depth,
        "bust_depth": bust_depth,
        "path": os.path.abspath('output/Output-Skeleton.jpg'),
        "time": "{:.3f}".format(time.time() - t)
    }
    return side_measures
    
#get_front_measurements("test_images/front/leo_test.jpeg", 175, True)
get_side_measurements("test_images/side/leo_test2_right.jpg", 181, True)
