import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_load import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
from models import Net
'''def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')'''

def showpoints(image, keypoints):
    plt.figure()

    keypoints = keypoints.data.numpy()
    keypoints = keypoints * 60.0 + 56
    keypoints = np.reshape(keypoints, (56, -1))
    print(keypoints)
    #plt.imshow(image)
    #plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, marker='.', c='r')
net = Net()

## TODO: load the best saved model parameters (by your path name)


## You'll need to un-comment the line below and add the correct name for *your* saved model
net.load_state_dict(torch.load(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontaltrainedmodel.pth'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv',
                                             root_dir=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat',
                                             transform=data_transform)
# load training data in batches
batch_size = 10

train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
for batch_i, data in enumerate(train_loader):
    # get the input images and their corresponding labels
    images = data['image']
    orig=images#cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    images = images.type(torch.FloatTensor)
    output_pts = net(images)
    print('orig', data['keypoints'])
    plt.figure(figsize=(5, 5))
    #showpoints(np.transpose(images[0].numpy(), (1,2,0)),output_pts)
    #showpoints(np.transpose(orig[0].numpy(),(0,1,2)),output_pts)
    showpoints(orig[0], output_pts)
    plt.show()
    break;
#key_pts_frame = pd.read_csv(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv')
'''n = 0
image_name = key_pts_frame.iloc[n, 0]
#key_pts = key_pts_frame.iloc[n, 1:].values.as_matrix()
key_pts = key_pts_frame.iloc[n, 1:].values
key_pts = key_pts.astype('float').reshape(-1, 2)
predicted=net(image_name)
print(key_pts)
print('Image name: ', image_name)
print('Landmarks shape: ', key_pts.shape)
print('First 4 key pts: {}'.format(key_pts[:4]))'''




#plt.figure(figsize=(5, 5))
#show_keypoints(mpimg.imread(os.path.join(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat', image_name)), key_pts)
#plt.show()