import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NetLL(nn.Module):

    def __init__(self):
        super(NetLL, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # obejctive is to bring down the image size to single unit-->
        # here given image size is 224x224px
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        # 429--> 215
        self.pool1 = nn.MaxPool2d(2, 2)
        # 215-->105 ...(32,110,110)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 105--> 105
        self.pool2 = nn.MaxPool2d(2, 2)
        # 105-->52

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 52-->52
        self.pool3 = nn.MaxPool2d(2, 2)
        # 52/2=26

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 26-->26
        self.pool4 = nn.MaxPool2d(2, 2)
        # 26/2=13

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        # 13-->13
        self.pool5 = nn.MaxPool2d(2, 2)
        # 13-->6

        # 6x6x512
        self.fc1 = nn.Linear(6 * 6 * 512, 1024)
        #         self.fc2 = nn.Linear(1024,1024)
        #self.fc2 = nn.Linear(1024, 112)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,256)
        #self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 112)
        self.fc5 = nn.Linear(112,58)
        #self.fc3 = nn.Linear(112,112)
        #self.fc4 = nn.Linear(112,112)

        '''self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        self.fc2_drop = nn.Dropout(p=.5)'''

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        '''x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.fc2(x)'''

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc3(x))
        x = self.fc5(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x