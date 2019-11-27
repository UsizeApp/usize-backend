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
import time
import datetime
from sklearn.model_selection import train_test_split

normal_key_pts_frame = pd.read_csv('extra_dataset/side_images_extra_dataset_11_keypoints.csv')
normal_data = normal_key_pts_frame.values[0]
person_id = normal_data[0]
keypoints = normal_data[1:]
normal_image = Image.open("extra_dataset/side/normal/" + str(int(person_id)) + "_side.jpg")

center = (normal_image.size[0]/2,normal_image.size[1]/2)

mirrored_key_pts_frame = pd.read_csv('extra_dataset/side_images_extra_dataset_11_keypointsREFLECTED.csv')
mirrored_data = mirrored_key_pts_frame.values[0]
mirrored_person_id = mirrored_data[0]
mirrored_keypoints = mirrored_data[1:]
mirrored_image = Image.open("extra_dataset/side/mirrored_noise/augmented_image_mirrored_noise_" + str(int(person_id)) + ".png")

plt.figure(1, figsize=(20,10))
plt.subplot(211)
plt.imshow(normal_image)
plt.scatter(center[0], center[1], s=20, marker='.', c='orange')
plt.scatter(keypoints[::2], keypoints[1::2], s=20, marker='.', c='m')
plt.subplot(212)
plt.imshow(mirrored_image)
plt.scatter(mirrored_keypoints[::2], mirrored_keypoints[1::2], s=20, marker='.', c='m')
plt.show()
