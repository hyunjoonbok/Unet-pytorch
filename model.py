# Create U-net Model / Layers 

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import os
import random

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Claim the layers that are needed in Init
        def CBR2d(in_channels, out_channels, kernel_size = 3,
                  stride = 1, padding = 1, bias = True):  #Convolutional, Batch-Normalization, Relu, and 2d
            layers = []
            # 1. Convolutional Layer
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                 kernel_size = kernel_size, stride = stride, padding = padding,
                                 bias = bias)]
            # 2. Batch-Normalization Layer
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            
            # 3. Relu Layer
            cbr = nn.Sequential(*layers)
            
            return cbr
        
        # Encoder Path (downsampling)
        
        # "kernel_size = 3, stride = 1, padding = 1, bias = True" can be skipped as it's already predefined above
        # First block at the first stage        
        self.enc1_1 = CBR2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        # Second block at the first stage
        self.enc1_2 = CBR2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        # Max-poolding 2x2 layer
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        
        # First block at the Second stage 
        self.enc2_1 = CBR2d(in_channels = 64, out_channels = 128)
        self.enc2_2 = CBR2d(in_channels = 64, out_channels = 128) 

        # Max-poolding 2x2 layer
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        
        # First block at the Third stage 
        self.enc3_1 = CBR2d(in_channels = 128, out_channels = 256)
        # Second block at the Third stage
        self.enc3_2 = CBR2d(in_channels = 128, out_channels = 256) 
               
        # Max-poolding 2x2 layer
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        
        # First block at the Fourth stage 
        self.enc4_1 = CBR2d(in_channels = 256, out_channels = 512)
        # Second block at the Fourth stage
        self.enc4_2 = CBR2d(in_channels = 256, out_channels = 512) 
               
        # Max-poolding 2x2 layer
        self.pool4 = nn.MaxPool2d(kernel_size = 2)   
        
        # First block at the Fifth stage 
        self.enc5_1 = CBR2d(in_channels = 512, out_channels = 1024)               
        
            
        # Decoder Path (upsampling)
        self.dec5_1 = CBR2d(in_channels = 1024, out_channels = 512)  
        
        # Up-convolution 2x2
        self.unpool4 = nn.ConvTranspose2d(in_channels = 512, out_channels = 512,
                                         kernel_size = 2, stride = 2, padding = 0, bias = True)
 
        # Second block at the Fourth stage       
        self.dec4_2 = CBR2d(in_channels = 2*512, out_channels = 512)
        # First block at the Fourth stage 
        self.dec4_1 = CBR2d(in_channels = 512, out_channels = 256)
        
        # Up-convolution 2x2
        self.unpool3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256,
                                         kernel_size = 2, stride = 2, padding = 0, bias = True)
        # Second block at the Third stage       
        self.dec3_2 = CBR2d(in_channels = 2*256, out_channels = 256)
        # First block at the Third stage 
        self.dec3_1 = CBR2d(in_channels = 256, out_channels = 128)

        # Up-convolution 2x2
        self.unpool2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128,
                                         kernel_size = 2, stride = 2, padding = 0, bias = True)
        # Second block at the Second stage       
        self.dec2_2 = CBR2d(in_channels = 2*128, out_channels = 128)
        # First block at the Second stage 
        self.dec2_1 = CBR2d(in_channels = 128, out_channels = 64)
        
        # Up-convolution 2x2
        self.unpool1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64,
                                         kernel_size = 2, stride = 2, padding = 0, bias = True)
        # Second block at the First stage       
        self.dec1_2 = CBR2d(in_channels = 2*64, out_channels = 64)
        # First block at the First stage 
        self.dec1_1 = CBR2d(in_channels = 64, out_channels = 64)        

        # Output layer   (1 x 1 Conv Layer)
        self.fc = nn.Conv2d(in_channels = 64, out_channels = 1,
                            kernel_size = 1, stride = 1, padding = 0, bias = True)
    
    
    def forward(self, x): # x is a input image
        # 1. Connect the Encoder part
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_1(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_1(enc2_1)
        pool2 = self.pool1(enc2_2)
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_1(enc3_1)
        pool3 = self.pool1(enc3_2)
        
        enc4_1 = self.enc1_1(pool2)
        enc4_2 = self.enc1_1(enc4_1)
        pool4 = self.pool1(enc4_2)
         
        enc5_1 = self.enc1_1(pool4)
        
        # 2. Connect the Decoder part
        dec5_1 = self.dec5_1(enc5_1) 
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1) # dim = [0: batch, 1: channel, 2: height, 3: width] each direction
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1) 
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1) 
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1) 
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.fc(dec1_1)
        
        return x
        