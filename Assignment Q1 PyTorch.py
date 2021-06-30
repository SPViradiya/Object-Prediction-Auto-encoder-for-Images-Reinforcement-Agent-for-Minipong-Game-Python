#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:55:20 2020

@author: spviradiya
"""

import pandas as pd
import numpy as np
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import timeit

# Loading the data
x_train = pd.read_csv("trainingpix.csv",header=None)
y_train = pd.read_csv("traininglabels.csv",header=None)
x_test = pd.read_csv("testingpix.csv",header=None)
y_test= pd.read_csv("testinglabels.csv",header=None)

# Preparing the TRAIN data, Reshaping it and making it a tensor
x_train_tensor = torch.tensor(np.reshape(x_train.to_numpy(),(x_train.shape[0],1,15,15))).float()
y_train_tensor = torch.tensor(y_train.to_numpy()).long()

#x_y_train_tensor_encoded = torch.zeros(len(x_y_train_tensor), x_y_train_tensor.max()).scatter_(1, x_y_train_tensor.unsqueeze(1)-1, 1.)

# Preparing the TEST data, Reshaping it and making it a tensor
x_test_tensor = torch.tensor(np.reshape(x_test.to_numpy(),(x_test.shape[0],1,15,15))).float()
y_test_tensor = torch.tensor(y_test.to_numpy()).long()

### Question 1 ###

########################################################################
# Define a Convolution Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # CNN
        self.conv1 = nn.Conv2d(1, 32, 3)        
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Fully connected network
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 14) # Predict x
        self.fc3 = nn.Linear(128, 14) # Predict y
        self.fc4 = nn.Linear(128, 5) # Predict z
        
        # Dropout layer
        self.dropout1 = nn.Dropout(0.25)

    def forward(self, x):
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)

        x_final = self.fc1(x)
        
        
        x = self.fc2(x_final) # x prediction
        y = self.fc3(x_final) # y prediction
        z = self.fc4(x_final) # z prediction
        
        
        return {"x":x,"y":y,"z":z}

net = Net()


#--------------------------------------------------------------
# Set the loss function

criterion = nn.CrossEntropyLoss()

# Setting SGD as optimizer
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)#, weight_decay = 0.01)

# Setting Adam as optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)

start = timeit.default_timer()
training_loss = []
# SGD required 170 epochs and gave 98% accuracy whereas Adam gave 98-99% with 50 epochs only
for epoch in range(50):  # loop over the dataset multiple times
    net.train() # Training the model
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(x_train_tensor.data)
    loss_x = criterion(outputs['x'], y_train_tensor[:,0])
    loss_y = criterion(outputs['y'], y_train_tensor[:,1])
    loss_z = criterion(outputs['z'], y_train_tensor[:,2])
    
    # Calculating total loss
    total_loss = loss_x + loss_y +   loss_z
    training_loss.append(total_loss)
    total_loss.backward()
    optimizer.step()

    # printing loss after every 10 epochs
    if epoch % 10 == 0:
        print('[%5d] loss: %.3f' % (epoch + 1, total_loss))
        
print('\nFinished Training...\n')
stop = timeit.default_timer()
timeTaken = stop - start
print('Time to run the training: ', timeTaken)
print('\n')

# Training accuracy
outputs = net(x_train_tensor.data)
_, x_predicted = torch.max(outputs['x'], 1)
_, y_predicted = torch.max(outputs['y'], 1)
_, z_predicted = torch.max(outputs['z'], 1)

x_acc_train = accuracy_score(y_train_tensor[:,0], x_predicted)
y_acc_train = accuracy_score(y_train_tensor[:,1], y_predicted)
z_acc_train = accuracy_score(y_train_tensor[:,2], z_predicted)
        
print('Training accuracy is x : %d %%' % (100 * x_acc_train))
print('Training accuracy is y : %d %%' % (100 * y_acc_train))
print('Training accuracy is z : %d %%' % (100 * z_acc_train))

print("----------------------------")

# Testing accuracy
outputs = net(x_test_tensor.data)
_, x_predicted = torch.max(outputs['x'], 1)
_, y_predicted = torch.max(outputs['y'], 1)
_, z_predicted = torch.max(outputs['z'], 1)

x_acc_test = accuracy_score(y_test_tensor[:,0], x_predicted)
y_acc_test = accuracy_score(y_test_tensor[:,1], y_predicted)
z_acc_test = accuracy_score(y_test_tensor[:,2], z_predicted)
        
print('Testing accuracy is x : %d %%' % (100 * x_acc_test))
print('Testing accuracy is y : %d %%' % (100 * y_acc_test))
print('Testing accuracy is z : %d %%' % (100 * z_acc_test))

# Plotting the training loss
plt.plot(range(1,51), training_loss)
plt.show()
