import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from torchvision.transforms import CenterCrop

from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
#from models import Net
from model_rightfoot import NetRF
#from model_429_vgg import Net
import torch.nn as nn
import torch.nn.functional as F
net = NetRF()

## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop

# testing that you've defined a transform

data_transform = transforms.Compose([#Rescale(250),
                                     #RandomCrop(224),
                                     #Rescale(225),
                                     RandomCrop(429),
                                     #CenterCrop([244,244]),
                                     Normalize(),
                                     ToTensor()])
assert(data_transform is not None), 'Define a data_transform'
transformed_dataset = FacialKeypointsDataset(csv_file=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontalpoints.csv',
                                             root_dir=r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\ceasar_mat_first100',
                                             transform=data_transform)
# load training data in batches
batch_size = 2

train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(params = net.parameters(), lr = 0.0001)


def train_net(n_epochs):
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            key_pts = key_pts[:,31:37]
            #print(key_pts)
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            #print(key_pts[:,16:42])
            #key_pts = key_pts[:, 16:42]
            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)
            # output_pts = output_pts.type(torch.FloatTensor)
            # print(output_pts.type)
            # print(key_pts.type)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

    print('Finished Training')

    # train your network


n_epochs = 100  # start small, and increase when you've decided on your model structure and hyperparams

# this is a Workspaces-specific context manager to keep the connection
# alive while training your model, not part of pytorch
#with active_session():
train_net(n_epochs)
PATH = r'C:\Users\Semanti Basu\Documents\OneDrive_2020-02-19\3D Ceaser dataset\Image and point generation\Image and point generation\frontaltrainedmodel_rightfoot_100epochs_first100.pth'
torch.save(net.state_dict(), PATH)