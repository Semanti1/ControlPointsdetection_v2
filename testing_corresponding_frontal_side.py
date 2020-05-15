import csv
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
#from models import Net
'''from model_resize429 import Net
from model_leftfoot import NetLF
from model_rightfoot import NetRF
from model_legs import NetLL'''
#from model_429_vgg import Net
from model_side_new import Net
#net = NetLF()
#net.load_state_dict(torch.load(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontaltrainedmodel_leftfoot_multifc_100epochs.pth'))
net1 = Net()
net1.load_state_dict(torch.load(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\profiletrained_fc3_lr0001_100epochs.pth'))
## print out your net and prepare it for testing (uncomment the line below)
#net2 = NetLL()
#net2.load_state_dict(torch.load(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontaltrainedmodel_legs_200epochs.pth'))
#net3 = NetRF()
#net3.load_state_dict(torch.load(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontaltrainedmodel_rightfoot_100epochs.pth'))
#net.eval()
#net1.eval()
#net2.eval()'''
net1.eval()
data_transform = transforms.Compose([#Rescale(225),
                                     RandomCrop(429),
                                     Normalize(),
                                     ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\side_corresp2frontal_test.csv',
                                             root_dir=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat_test_random',
                                             transform=data_transform)
# load training data in batches
batch_size = 1

test_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

def net_sample_output():
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        #key_pts = key_pts[:, 31:37]
        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)
        '''output_pts_rf = net3(images)
        output_pts_rf = output_pts_rf.view(output_pts_rf.size()[0], 6, -1)
        output_pts_rf1 = net3(images)
        output_pts_rf1 = output_pts_rf1.view(output_pts_rf1.size()[0], 6, -1)
        output_pts_rf2 = net3(images)
        output_pts_rf2 = output_pts_rf2.view(output_pts_rf2.size()[0], 6, -1)
        output_pts_rf3 = net3(images)
        output_pts_rf3 = output_pts_rf3.view(output_pts_rf3.size()[0], 6, -1)
        output_pts_rf = (output_pts_rf + output_pts_rf1 + output_pts_rf2 + output_pts_rf3)/4.0
        # forward pass to get net output
        output_pts = net(images)
        outpts2 = net1(images)
        outpts2 = outpts2.view(outpts2.size()[0], 56, -1)
        outpts3 = net1(images)
        outpts3 = outpts3.view(outpts3.size()[0], 56, -1)
        outpts4 = net1(images)
        outpts4 = outpts4.view(outpts4.size()[0], 56, -1)
        out_wholebod = net1(images)
        out_wholebod = out_wholebod.view(out_wholebod.size()[0], 56, -1)
        out_wholebod = (out_wholebod + outpts2+outpts3+outpts4) * (1/4.0)
        #print(out_wholebod[:,20:26])
        out_leg = net2(images)
        out_leg = out_leg.view(out_leg.size()[0], 29, -1)
        out_leg2 = net2(images)
        out_leg2 = out_leg2.view(out_leg2.size()[0], 29, -1)
        out_leg3 = net2(images)
        out_leg3 = out_leg3.view(out_leg3.size()[0], 29, -1)
        out_leg4 = net2(images)
        out_leg4 = out_leg4.view(out_leg4.size()[0], 29, -1)
        out_leg = (out_leg + out_leg2 + out_leg3 + out_leg4) * (1/4.0)
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 6, -1)
        output_pts2 = net(images)
        output_pts2 = output_pts2.view(output_pts2.size()[0], 6, -1)
        output_pts3 = net(images)
        output_pts3 = output_pts3.view(output_pts3.size()[0], 6, -1)
        output_pts4 = net(images)
        output_pts4 = output_pts4.view(output_pts4.size()[0], 6, -1)
        output_pts = (output_pts + output_pts2 + output_pts3 + output_pts4)/4.0
        out_wholebod[:, 20:26] = (1.0/2) * (out_wholebod[:, 20:26] + output_pts)
        out_wholebod[:, 14:43] = (1.0 / 2) * (out_wholebod[:, 14:43] + out_leg)
        out_wholebod[:, 31:37] = (1.0 / 2) * (out_wholebod[:, 31:37] + output_pts_rf)'''
        output_pts = net1(images)
        output_pts = output_pts.view(output_pts.size()[0], 26, -1)
        #print(output_pts)
        print(sample['name'])
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
            #return images, out_wholebod, key_pts


def show_all_keypoints(image, predicted_key_pts, gt_pts=None,lftpts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.imsave(r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\test_image_discrepancy.jpg',image)

    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='x', c='m')
    predicted_key_pts[:, 0] = predicted_key_pts[:, 0] + 35
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='o', c='g')
    if lftpts is not None:
        plt.scatter(lftpts[:, 0], lftpts[:, 1], s=20, marker='*', c='b')
    with open("test_discrepancy_side.csv", "w", newline='') as f:
        writer = csv.writer(f)
        gt_pts = gt_pts.numpy()
        writer.writerow(predicted_key_pts.flatten())
        writer.writerow(gt_pts.flatten())


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, onlyfoot=None, batch_size=1):
    for i in range(batch_size):
        plt.figure(figsize=(20, 20))
        #ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data  # get the image from it's Variable wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        #print(predicted_key_pts.size())
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100
        print(predicted_key_pts.shape)
        #print(predicted_key_pts)
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100
        if onlyfoot is not None:
            leftfoot_pts = onlyfoot[i].data
            # print(predicted_key_pts.size())
            leftfoot_pts = leftfoot_pts.numpy()
            #leftfoot_pts = onlyfoot[i]
            leftfoot_pts = leftfoot_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()




# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())

# call it
visualize_output(test_images, test_outputs, gt_pts)