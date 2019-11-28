from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
from scipy.spatial import distance as dist
import time
import os
import math

MOVE = 0

def plot_line(p1,p2,color = 'orange'):
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]], c = color)
    return

def midpoint(p1, p2):
    if (p1 == None or p2 == None):
        return -1
    return [int((p1[0]+p2[0])/2) , int((p1[1]+p2[1])/2)]

def calculate_ellipse_perimeter(minor_axis,mayor_axis):
    a = minor_axis
    b = mayor_axis
    perimeter = math.pi * ( 3*(a+b) - math.sqrt( (3*a + b) * (a + 3*b) ) )
    return perimeter

def set_body_points(frontal_points, lateral_points):
    body = {
        "frontal_points" : {
            "right_wrist" : [frontal_points[0],frontal_points[1]],
            "right_elbow" : [frontal_points[2],frontal_points[3]],
            "right_shoulder1" : [frontal_points[4],frontal_points[5]],
            "right_shoulder2" : [frontal_points[6],frontal_points[7]],
            "head_right" : [frontal_points[8],frontal_points[9]],
            "head_left" : [frontal_points[10],frontal_points[11]],
            "left_shoulder2" : [frontal_points[12],frontal_points[13]],
            "left_shoulder1" : [frontal_points[14],frontal_points[15]],
            "left_elbow" : [frontal_points[16],frontal_points[17]],
            "left_wrist" : [frontal_points[18],frontal_points[19]],
            "left_chest" : [frontal_points[20],frontal_points[21]],
            "left_bust" : [frontal_points[22],frontal_points[23]],
            "left_waist" : [frontal_points[24],frontal_points[25]],
            "left_hips" : [frontal_points[26],frontal_points[27]],
            "left_knee" : [frontal_points[28],frontal_points[29]],
            "outer_left_ankle" : [frontal_points[30],frontal_points[31]],
            "outer_right_ankle" : [frontal_points[32],frontal_points[33]],
            "right_knee" : [frontal_points[34],frontal_points[35]],
            "right_hips" : [frontal_points[36],frontal_points[37]],
            "right_waist" : [frontal_points[38],frontal_points[39]],
            "right_bust" : [frontal_points[40],frontal_points[41]],
            "right_chest" : [frontal_points[42],frontal_points[43]]
        },
        "lateral_points": {
            "chest" : [lateral_points[0],lateral_points[1]],
            "upper_bust" : [lateral_points[2],lateral_points[3]],
            "lower_bust" : [lateral_points[4],lateral_points[5]],
            "frontal_waist" : [lateral_points[6],lateral_points[7]],
            "frontal_hips" : [lateral_points[8],lateral_points[9]],
            "heel" : [lateral_points[10],lateral_points[11]],
            "back_hips" : [lateral_points[12],lateral_points[13]],
            "back_waist" : [lateral_points[14],lateral_points[15]],
            "lower_back" : [lateral_points[16],lateral_points[17]],
            "upper_back" : [lateral_points[18],lateral_points[19]],
            "head" : [lateral_points[20],lateral_points[21]]
        }
    }
    return body

def calculate(body_part, body, frontal_pixelsPerMetric, lateral_pixelsPerMetric):

    if body_part == "right_arm":
        right_humerus = dist.euclidean(body["frontal_points"]["right_shoulder1"], body["frontal_points"]["right_elbow"])
        right_forearm = dist.euclidean(body["frontal_points"]["right_elbow"], body["frontal_points"]["right_wrist"])
        measure = (right_humerus + right_forearm) / frontal_pixelsPerMetric
    elif body_part == "left_arm":
        left_humerus = dist.euclidean(body["frontal_points"]["left_shoulder1"], body["frontal_points"]["left_elbow"])
        left_forearm = dist.euclidean(body["frontal_points"]["left_elbow"], body["frontal_points"]["left_wrist"])
        measure = (left_humerus + left_forearm) / frontal_pixelsPerMetric
    elif body_part == "right_leg":
        right_femur = dist.euclidean(body["frontal_points"]["right_hips"], body["frontal_points"]["right_knee"])
        right_tibia = dist.euclidean(body["frontal_points"]["right_knee"], body["frontal_points"]["outer_right_ankle"])
        measure = (right_femur + right_tibia) / frontal_pixelsPerMetric
    elif body_part == "left_leg":
        left_femur = dist.euclidean(body["frontal_points"]["left_hips"], body["frontal_points"]["left_knee"])
        left_tibia = dist.euclidean(body["frontal_points"]["left_knee"], body["frontal_points"]["outer_left_ankle"])
        measure = (left_femur + left_tibia) / frontal_pixelsPerMetric
    elif body_part == "waist":
        frontal_waist = dist.euclidean(body["frontal_points"]["right_waist"], body["frontal_points"]["left_waist"]) / frontal_pixelsPerMetric
        lateral_waist = dist.euclidean(body["lateral_points"]["frontal_waist"], body["lateral_points"]["back_waist"]) / lateral_pixelsPerMetric
        measure = calculate_ellipse_perimeter(frontal_waist/2,lateral_waist/2)
    elif body_part == "hips":
        frontal_hips = dist.euclidean(body["frontal_points"]["right_hips"], body["frontal_points"]["left_hips"]) / frontal_pixelsPerMetric 
        lateral_hips = dist.euclidean(body["lateral_points"]["frontal_hips"], body["lateral_points"]["back_hips"]) / lateral_pixelsPerMetric
        measure = calculate_ellipse_perimeter(frontal_hips/2,lateral_hips/2)
    elif body_part == "chest":
        frontal_chest = dist.euclidean(body["frontal_points"]["right_chest"], body["frontal_points"]["left_chest"]) / frontal_pixelsPerMetric 
        lateral_chest = dist.euclidean(body["lateral_points"]["chest"], body["lateral_points"]["upper_back"]) / lateral_pixelsPerMetric
        measure = calculate_ellipse_perimeter(frontal_chest/2,lateral_chest/2) 
    elif body_part == "bust":
        frontal_bust = dist.euclidean(body["frontal_points"]["right_bust"], body["frontal_points"]["left_bust"]) / frontal_pixelsPerMetric 
        lateral_bust = dist.euclidean(body["lateral_points"]["lower_bust"], body["lateral_points"]["lower_back"]) / lateral_pixelsPerMetric
        measure = calculate_ellipse_perimeter(frontal_bust/2,lateral_bust/2) 

    return measure

def move(frame, point, direction, n, color):
    #0 move up, 1 move right, 2 move down, 3 move left
    #cv2.circle(frame, (point[0],point[1]), 8, (0,125,0), thickness=-1, lineType=cv2.FILLED)
    final = [int(point[1]),int(point[0])]
    if((frame[tuple(final)] == color).all()):
        print("No move")
        return [final[1],final[0]]
    if(direction == 0):
        for i in range(n):
            final[0] = final[0] - 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 1):
        for i in range(n):
            final[1] = final[1] + 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 2):
        for i in range(n):
            final[0] = final[0] + 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 3):
        for i in range(n):
            final[1] = final[1] - 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 4):
        for i in range(n):
            final[0] = final[0] - 1 
            final[1] = final[1] + 1  
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 5):
        for i in range(n):
            final[0] = final[0] + 1 
            final[1] = final[1] + 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 6):
        for i in range(n):
            final[0] = final[0] + 1 
            final[1] = final[1] - 1 
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    if(direction == 7):
        for i in range(n):
            final[0] = final[0] - 1 
            final[1] = final[1] - 1  
            if((frame[tuple(final)] == color).all()):
                print("i:",i)
                return [final[1],final[0]]
    return [final[1],final[0]]

nn_image_width = 160
nn_image_height = 160

def transform_image(image,width,height,channels):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges_channel = cv2.Canny(frame, int(max(0, np.mean(frame))), int(min(255, np.mean(frame))))
    edges_channel = np.expand_dims(edges_channel,axis = 2)
    feeded_image = np.asarray(image)
    feeded_image = np.concatenate((feeded_image,edges_channel),axis = 2)
    feeded_image = cv2.resize(feeded_image,(nn_image_width,nn_image_height))
    feeded_image = feeded_image/255
    feeded_image = feeded_image.reshape(1,nn_width,nn_height,channels)
    return feeded_image

nn_width = 160
nn_height = 160

def get_measurements(frontal_file, lateral_file, body_height_cm, show = False):
    #Transform images to feed the model
    original_frontal_image = Image.open(frontal_file)
    original_lateral_image = Image.open(lateral_file)
    frontal_width, frontal_height = original_frontal_image.size
    lateral_width, lateral_height = original_lateral_image.size
    #print(frontal_width,frontal_height)
    feeded_frontal_image = transform_image(original_frontal_image, nn_width, nn_height,4)
    feeded_lateral_image = transform_image(original_lateral_image, nn_width, nn_height,4)

    t = time.time()
    
    #Load Models
    frontal_model = keras.models.load_model('../models/(BEST_FRONT)UsizeNetConvolutional_front_4-channels_1000-epochs_2019-11-02 06_08_57.h5')    
    lateral_model = keras.models.load_model('../models/(BEST_SIDE)UsizeNetConvolutional_side_4-channels_100-epochs_2019-11-23 19_15_11.h5')  
    
    #Predict Keypoints
    predicted_frontal_keypoints = frontal_model.predict(feeded_frontal_image)
    predicted_frontal_keypoints = predicted_frontal_keypoints[0]
    predicted_lateral_keypoints = lateral_model.predict(feeded_lateral_image)
    predicted_lateral_keypoints = predicted_lateral_keypoints[0]

    #Rescale keypoints
    predicted_frontal_keypoints[::2] = np.round(predicted_frontal_keypoints[::2] * (frontal_width/nn_width))
    predicted_frontal_keypoints[1::2] = np.round(predicted_frontal_keypoints[1::2] * (frontal_height/nn_height))
    predicted_lateral_keypoints[::2] = np.round(predicted_lateral_keypoints[::2] * (lateral_width/nn_width))
    predicted_lateral_keypoints[1::2] = np.round(predicted_lateral_keypoints[1::2] * (lateral_height/nn_height))

    #Set body points
    body = set_body_points(predicted_frontal_keypoints, predicted_lateral_keypoints)

    #Move Keypoints---------------------------------------------------------------------------------------------------

    if MOVE:

        frontal_frame = cv2.imread(frontal_file)
        lateral_frame = cv2.imread(lateral_file)
        frontal_frame_contours = np.copy(frontal_frame)
        lateral_frame_contours = np.copy(lateral_frame)
        frontal_edged = cv2.Canny(frontal_frame, int(max(0, np.mean(frontal_frame))), int(min(255, np.mean(frontal_frame))))
        lateral_edged = cv2.Canny(lateral_frame, int(max(0, np.mean(lateral_frame))), int(min(255, np.mean(lateral_frame))))
        frontal_contours, _ = cv2.findContours(frontal_edged,cv2.RETR_TREE,cv2.cv2.CHAIN_APPROX_NONE)
        lateral_contours, _ = cv2.findContours(lateral_edged,cv2.RETR_TREE,cv2.cv2.CHAIN_APPROX_NONE)
        frontal_frame_contours = cv2.drawContours(frontal_frame_contours, frontal_contours, -1, (0,255,0), 3)
        lateral_frame_contours = cv2.drawContours(lateral_frame_contours, lateral_contours, -1, (0,255,0), 3)

        #Move frontal points
        body['frontal_points']['right_shoulder1'] = move(frontal_frame_contours,body['frontal_points']['right_shoulder1'],1,500, (0,255,0))
        body['frontal_points']['right_shoulder2'] = move(frontal_frame_contours,body['frontal_points']['right_shoulder2'],1,500, (0,255,0))
        body['frontal_points']['left_shoulder1'] = move(frontal_frame_contours,body['frontal_points']['left_shoulder1'],3,500, (0,255,0))
        body['frontal_points']['left_shoulder2'] = move(frontal_frame_contours,body['frontal_points']['left_shoulder2'],3,500, (0,255,0))
        

        #Move lateral points
        #print("<<<<{}>>>>".format(body['lateral_points']['chest']))
        body['lateral_points']['chest'] = move(lateral_frame_contours,body['lateral_points']['chest'],3,500, (0,255,0))
        #print("<<<<{}>>>>".format(body['lateral_points']['chest']))
        body['lateral_points']['upper_back'] = move(lateral_frame_contours,body['lateral_points']['upper_back'],1,500, (0,255,0))
        body['lateral_points']['lower_back'] = move(lateral_frame_contours,body['lateral_points']['lower_back'],1,500, (0,255,0))
        body['lateral_points']['back_hips'] = move(lateral_frame_contours,body['lateral_points']['back_hips'],1,500, (0,255,0))
        body['lateral_points']['frontal_hips'] = move(lateral_frame_contours,body['lateral_points']['frontal_hips'],3,500, (0,255,0))
        body['lateral_points']['upper_bust'] = move(lateral_frame_contours,body['lateral_points']['upper_bust'],3,500, (0,255,0))
        body['lateral_points']['lower_bust'] = move(lateral_frame_contours,body['lateral_points']['lower_bust'],3,500, (0,255,0))

        #Plot changes
        '''if show:
            plt.imshow(original_frontal_image)
            plt.scatter(predicted_frontal_keypoints[::2], predicted_frontal_keypoints[1::2], s=20, marker='.', c='m')
            plt.scatter(body['frontal_points']['right_shoulder1'][0], body['frontal_points']['right_shoulder1'][1], s=30, c = 'b')
            plt.scatter(body['frontal_points']['right_shoulder2'][0], body['frontal_points']['right_shoulder2'][1], s=30, c = 'b')
            plt.scatter(body['frontal_points']['left_shoulder1'][0], body['frontal_points']['left_shoulder1'][1], s=30, c = 'b')
            plt.scatter(body['frontal_points']['left_shoulder2'][0], body['frontal_points']['left_shoulder2'][1], s=30, c = 'b')
            plt.show()

            plt.imshow(original_lateral_image)
            plt.scatter(predicted_lateral_keypoints[::2], predicted_lateral_keypoints[1::2], s=20, marker='.', c='m')
            plt.scatter(body['lateral_points']['chest'][0], body['lateral_points']['chest'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['upper_back'][0], body['lateral_points']['upper_back'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['lower_back'][0], body['lateral_points']['lower_back'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['back_hips'][0], body['lateral_points']['back_hips'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['frontal_hips'][0], body['lateral_points']['frontal_hips'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['upper_bust'][0], body['lateral_points']['upper_bust'][1], s=30, c = 'b')
            plt.scatter(body['lateral_points']['lower_bust'][0], body['lateral_points']['lower_bust'][1], s=30, c = 'b')
            #plt.show()'''

        '''cv2.imwrite(os.path.abspath('output/front_contours.jpg'), frontal_frame_contours)
        cv2.imwrite(os.path.abspath('output/lateral_contours.jpg'), lateral_frame_contours)'''

    #-----------------------------------------------------------------------------------------------------------------

    #Extra frontal points:
    floor = midpoint(body["frontal_points"]["outer_left_ankle"],body["frontal_points"]["outer_right_ankle"])
    head_upper = midpoint(body["frontal_points"]["head_right"],body["frontal_points"]["head_left"])

    #Frontal height:
    frontal_height_in_pixels = dist.euclidean(floor, head_upper)
    frontal_pixelsPerMetric = frontal_height_in_pixels/body_height_cm

    #Lateral height:
    lateral_height_in_pixels = dist.euclidean(body["lateral_points"]["heel"], body["lateral_points"]["head"])
    lateral_pixelsPerMetric = lateral_height_in_pixels/body_height_cm

    #Calculate lengths:
    right_arm   = calculate("right_arm",  body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    left_arm    = calculate("left_arm",   body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    right_leg   = calculate("right_leg",  body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    left_leg    = calculate("left_leg",   body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    waist       = calculate("waist",      body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    hips        = calculate("hips",       body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    chest       = calculate("chest",      body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)
    bust        = calculate("bust",       body, frontal_pixelsPerMetric, lateral_pixelsPerMetric)

    measures = {
        "right_arm": right_arm,
        "left_arm": left_arm,
        "right_leg": right_leg,
        "left_leg": left_leg,
        "waist_length": waist,
        "hips_length": hips,
        "chest_length": chest,
        "bust_length": bust,
        "path": os.path.abspath('output/Output-Skeleton.jpg'),
        "time": "{:.3f}".format(time.time() - t)
    }

    if show:

        #Frontal image plot:
        plt.subplot(1, 2, 1)
        plt.imshow(original_frontal_image)
        plt.scatter(predicted_frontal_keypoints[::2], predicted_frontal_keypoints[1::2], s=20, marker='.', c='m')
        #plt.scatter(original_keypoints[1::2], original_keypoints[::2], s=20, marker='.', c='orange')
        plot_line(floor,head_upper)
        plot_line(body["frontal_points"]["right_hips"],body["frontal_points"]["right_knee"])
        plot_line(body["frontal_points"]["right_knee"],body["frontal_points"]["outer_right_ankle"])
        plot_line(body["frontal_points"]["left_hips"],body["frontal_points"]["left_knee"])
        plot_line(body["frontal_points"]["left_knee"],body["frontal_points"]["outer_left_ankle"])
        plot_line(body["frontal_points"]["right_shoulder1"],body["frontal_points"]["right_elbow"])
        plot_line(body["frontal_points"]["right_elbow"],body["frontal_points"]["right_wrist"])
        plot_line(body["frontal_points"]["left_shoulder1"],body["frontal_points"]["left_elbow"])
        plot_line(body["frontal_points"]["left_elbow"],body["frontal_points"]["left_wrist"])
        plot_line(body["frontal_points"]["right_hips"],body["frontal_points"]["left_hips"])
        plot_line(body["frontal_points"]["right_chest"],body["frontal_points"]["left_chest"])
        plot_line(body["frontal_points"]["right_bust"],body["frontal_points"]["left_bust"])

        #Lateral image plot
        plt.subplot(1, 2, 2)
        plt.imshow(original_lateral_image)
        plt.scatter(predicted_lateral_keypoints[::2], predicted_lateral_keypoints[1::2], s=20, marker='.', c='m')
        plot_line(body["lateral_points"]["heel"],body["lateral_points"]["head"])
        plot_line(body["lateral_points"]["frontal_hips"], body["lateral_points"]["back_hips"])
        plot_line(body["lateral_points"]["chest"], body["lateral_points"]["upper_back"])
        plot_line(body["lateral_points"]["upper_bust"], body["lateral_points"]["lower_back"])
        plt.savefig(os.path.abspath('output/Skeleton-Keypoints.jpg'))
        plt.show()

    return measures

    
#measures = get_measurements("test_images/front/leo_test.jpeg","test_images/side/leo_test2_right.jpg", 181, True)
#print("Medidas finales:")
#for key,value in measures.items():
#   print("{}: {}".format(key,value))

