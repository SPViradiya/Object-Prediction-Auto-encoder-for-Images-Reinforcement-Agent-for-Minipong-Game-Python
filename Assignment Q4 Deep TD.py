#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:25:41 2020

@author: spviradiya
"""
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
from minipong import Minipong
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import random
import timeit
from itertools import count

def plot_res(values, title='', save = False, i = 0):   
    ''' Plot the reward curve and histogram of results over time.'''

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        pass
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    
    if i % 50 == 0:
        plt.show()
    
    plt.close()

class DQN(nn.Module):
    """Deep Q Neural Network class. """

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        # super() to call parent class initialisation
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        #print(x.shape)
        x = self.fc2(x)
        return x

    def update(self, state, y):
        y_pred = self(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self(torch.Tensor(state))


env = Minipong(level=3, size=5)

# Number of states
n_state = 3
# Number of actions
n_action = 3
# Number of episodes
episodes = 1000
# Number of hidden nodes in the DQN
n_hidden = 150
# Learning rate
lr = 0.001

epsilon=1
gamma  = 0.95
eps_decay = 0.99
final = []
running_reward = 0
log_interval = 50

start = timeit.default_timer()
dqn = DQN(n_state, n_action, n_hidden, lr)

for i_episode in count(1):
    state, ep_reward, done = env.reset(), 0, False
    
    for t in range(1, 10000):  # Don't infinite loop while learning
    
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.sampleaction()
            else:
                q_values = dqn.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)
            
            # Update total
            ep_reward += reward
            q_values = dqn.predict(state).tolist()
             
            if done:
                q_values[action] = reward
                # Update network weights
                dqn.update(state, q_values)
                break

            # Update network weights using the last step
            q_values_next = dqn.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            dqn.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(ep_reward)
        plot_res(final, "Deep TD", i = i_episode)
    
    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    # log results
    if i_episode % log_interval == 0:
        print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward))
    
    # check if we have solved minipong
    if running_reward > 300:
        print("Running reward is now {:.2f} and the last episode "
              "runs to {} time steps!".format(running_reward, t))
    

print('\nFinished Training...\n')
stop = timeit.default_timer()
timeTaken = stop - start
print('Time to run the training: ', timeTaken)
print('\n')