#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:08:31 2020

@author: spviradiya
"""


import pandas as pd
import numpy as np
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
#from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

# Loading the data
x_train = pd.read_csv("trainingpix.csv",header=None)
x_test = pd.read_csv("testingpix.csv",header=None)

# Preparing the TRAIN data, Reshaping it and making it a tensor
x_train_tensor = torch.tensor(np.reshape(x_train.to_numpy(),(x_train.shape[0],1,15,15))).float()

# Preparing the TEST data, Reshaping it and making it a tensor
x_test_tensor = torch.tensor(np.reshape(x_test.to_numpy(),(x_test.shape[0],1,15,15))).float()

### Question 2 ###

def create_img(img_name, data):
    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(data.reshape(15,15), origin='lower' , cmap=plt.cm.coolwarm)
    plt.savefig('q2_img/'+img_name+'.png', dpi=300)
    plt.show()
    
print_shape = 0

class MiniPongEncoder(nn.Module):
    def __init__(self):
        super(MiniPongEncoder, self).__init__()
        
        # CNN
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 2, padding= 1) # 15* 15

        #self.pool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        
        # Decoder
        self.iconv1 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.iconv2 = nn.ConvTranspose2d(16, 8, 3, stride = 1, padding = 1)
        self.iconv3 = nn.ConvTranspose2d(8, 1, 3, stride = 1, padding = 1)
        
        self.dropout1 = nn.Dropout(0.25)
        
    def encoder(self, x):
        if print_shape == 1:
            print("Encoder")
            print(x.shape)
        x = F.relu(self.conv1(x))
        
        if print_shape == 1:
            print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        
        #if print_shape == 1:
            #print(x.shape)
        #x = self.pool3(F.relu(self.conv3(x)))
        
        if print_shape == 1:
            print(x.shape)
        x = self.dropout1(x)
        return x
    
    def decoder(self, x):
        if print_shape == 1:
            print("Decoder")
            print(x.shape)
        x = F.relu(self.iconv1(x))
        
        if print_shape == 1:
            print(x.shape)
        x = F.relu(self.iconv2(x))
        
        if print_shape == 1:
            print(x.shape)
        x = torch.tanh(self.iconv3(x))
        
        if print_shape == 1:
            print(x.shape)
        return(x)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


criterion = nn.MSELoss()
auto_encoder = MiniPongEncoder()
optimizer = optim.Adam(auto_encoder.parameters(), lr=1e-3, weight_decay=1e-5)
training_loss = []

for epoch in range(150):  # loop over the dataset multiple times
    auto_encoder.train() # Training the model
    
    # forward + backward + optimize
    outputs = auto_encoder(x_train_tensor)
    loss = criterion(outputs, x_train_tensor)
    
    training_loss.append(loss)
    
    optimizer.zero_grad()
    # Calculating total loss
    loss.backward()
    optimizer.step()
    
    # printing loss after every 10 epochs
    if epoch % 9 == 0:
        print('[%5d] loss: %.3f' % (epoch + 1, loss))
   
     
index_img = 15
create_img("proto_"+str(index_img)+"_original",x_train_tensor[[[index_img]]])
otpt = auto_encoder(x_train_tensor[[[index_img]]])
create_img("proto_"+str(index_img)+"_encoded",otpt.data) 

index_img = 200
create_img("proto_"+str(index_img)+"_original",x_train_tensor[[[index_img]]])
otpt = auto_encoder(x_train_tensor[[[index_img]]])
create_img("proto_"+str(index_img)+"_encoded",otpt.data) 

index_img = 220
create_img("proto_"+str(index_img)+"_original",x_train_tensor[[[index_img]]])
otpt = auto_encoder(x_train_tensor[[[index_img]]])
create_img("proto_"+str(index_img)+"_encoded",otpt.data)

index_img = 202
create_img("proto_"+str(index_img)+"_original",x_train_tensor[[[index_img]]])
otpt = auto_encoder(x_train_tensor[[[index_img]]])
create_img("proto_"+str(index_img)+"_encoded",otpt.data)        
    
# Plotting the training loss
plt.plot(range(1,151), training_loss)
plt.show()
