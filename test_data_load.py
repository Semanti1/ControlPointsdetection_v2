# Construct the dataset
from data_load import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.image as mpimg
from torchvision import transforms
#from visualize import show_keypoints
def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')

face_dataset = FacialKeypointsDataset(csv_file=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv',
                                      root_dir=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat')

# print some stats about the dataset
print('Length of dataset: ', len(face_dataset))

num_to_display = 3
fig = plt.figure(figsize=(20, 10))
for i in range(num_to_display):
    # define the size of images
    #fig = plt.figure(figsize=(20, 10))
    #plt.figure(figsize=(20, 10))
    # randomly select a sample
    rand_i = np.random.randint(0, len(face_dataset))
    sample = face_dataset[rand_i]

    # print the shape of the image and keypoints
    print(i, sample['image'].shape, sample['keypoints'].shape)

    ax = plt.subplot(1, num_to_display, i + 1)
    ax.set_title('Sample #{}'.format(i))

    # Using the same display function, defined earlier
    show_keypoints(sample['image'], sample['keypoints'])
plt.show()

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# create the transformed dataset
'''transformed_dataset = FacialKeypointsDataset(csv_file='/data/training_frames_keypoints.csv',
                                             root_dir='/data/training/',
                                             transform=data_transform)'''
transformed_dataset = FacialKeypointsDataset(csv_file=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv',
                                             root_dir=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat',
                                             transform=data_transform)
print('Number of images: ', len(transformed_dataset))

# make sure the sample tensors are the expected size
for i in range(5):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())