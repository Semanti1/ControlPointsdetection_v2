# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

key_pts_frame = pd.read_csv(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv')
n = 0
image_name = key_pts_frame.iloc[n, 0]
#key_pts = key_pts_frame.iloc[n, 1:].values.as_matrix()
key_pts = key_pts_frame.iloc[n, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)
print(key_pts)
print('Image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
#print('First 4 key pts: {}'.format(key_pts[:4]))

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


plt.figure(figsize=(5, 5))
show_keypoints(mpimg.imread(os.path.join(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat', image_name)), key_pts)
plt.show()