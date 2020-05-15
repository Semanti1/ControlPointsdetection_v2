import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

import keras.utils

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Lambda, MaxPooling2D, ZeroPadding2D
from keras.models import Input, Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2
def VGG(input_shape):

    def conv_block(x, n_filters, block_num):
        x = Conv2D(n_filters, kernel_size=(3, 3), strides=1, padding='same',
                   kernel_regularizer=l2(5e-4), name="Conv" + str(block_num))(x)
        x = BatchNormalization(name="bN" + str(block_num))(x)
        x = Activation("relu", name="relu"+str(block_num))(x)

        return x

    input = Input(input_shape, name="Input")

    # Layers 1-2
    x = conv_block(input, n_filters=64, block_num=1)
    x = conv_block(x, n_filters=64, block_num=2)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='pool1')(x)

    # Layers 3-4
    x = conv_block(x, n_filters=128, block_num=3)
    x = conv_block(x, n_filters=128, block_num=4)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='pool2')(x)

    # Layers 5-7
    x = conv_block(x, n_filters=256, block_num=5)
    x = conv_block(x, n_filters=256, block_num=6)
    x = conv_block(x, n_filters=256, block_num=7)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='pool3')(x)

    # Layers 8-10
    x = conv_block(x, n_filters=512, block_num=8)
    x = conv_block(x, n_filters=512, block_num=9)
    x = conv_block(x, n_filters=512, block_num=10)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='pool4')(x)

    # Layers 11-13
    x = conv_block(x, n_filters=512, block_num=11)
    x = conv_block(x, n_filters=512, block_num=12)
    x = conv_block(x, n_filters=512, block_num=13)
    x = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid', name='pool5')(x)

    # Layers 14-16
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    # x = Dense(units=4096, activation='relu', kernel_regularizer=l2(5e-4), name="Dense1")(x)
    # x = Dropout(rate=0.5)(x)
    x = Dense(units=512, activation='relu', kernel_regularizer=l2(5e-4), name="Dense1")(x)

    # output
    output = Dense(units=2*56, activation='linear', kernel_regularizer=l2(5e-4), name="output")(x)

    model = Model(inputs=input, outputs=output, name="VGG")

    return model


vgg16 = VGG((429,429,1))
print("{}".format(vgg16.summary()))

'''class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # obejctive is to bring down the image size to single unit-->
        # here given image size is 224x224px
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 224--> 224-5+1=220
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2=110 ...(32,110,110)

        self.conv2 = nn.Conv2d(32, 64, 3)
        # 110--> 110-3+1=108
        self.pool2 = nn.MaxPool2d(2, 2)
        # 108/2=54

        self.conv3 = nn.Conv2d(64, 128, 3)
        # 54-->54-3+1=52
        self.pool3 = nn.MaxPool2d(2, 2)
        # 52/2=26

        self.conv4 = nn.Conv2d(128, 256, 3)
        # 26-->26-3+1=24
        self.pool4 = nn.MaxPool2d(2, 2)
        # 24/2=12

        self.conv5 = nn.Conv2d(256, 512, 1)
        # 12-->12-1+1=12
        self.pool5 = nn.MaxPool2d(2, 2)
        # 12/2=6

        # 6x6x512
        self.fc1 = nn.Linear(6 * 6 * 512, 1024)
        #         self.fc2 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024, 112)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        # self.fc2_drop = nn.Dropout(p=.5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x'''