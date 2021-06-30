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


""" 
Q1 answer about creating manual policy (no RL) that successfully plays pong
"""
def mypolicy(state):
    """
    State is 0: + is in the middle, do nothing
    State is negative : + is on the left, move left
    State is positive: + is on the right, move right
    
    0: do nothing
    1: move left
    2: move right
    """
    action = 0 # Do nothing
    
    if state > 0:
        action = 2
    elif state < 0:
        action = 1
    
    return action


"""
DQN Network
"""


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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, action_dim)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x):
        # x-> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> y
        #print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        #print(x.shape)
        x = F.leaky_relu(self.fc2(x))
        #print(x.shape)
        x = self.fc3(x)
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
        
def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=1, eps_decay=0.99,
               title = 'DQL', verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    for i, episode in enumerate(range(episodes), start = 1):        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        #print(state)
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.sampleaction()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)
            
            # Update total
            total += reward
            q_values = model.predict(state).tolist()
             
            if done:
                q_values[action] = reward
                # Update network weights
                model.update(state, q_values)
                break

            # Update network weights using the last step
            q_values_next = model.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.1)
        final.append(total)
        plot_res(final, title, i = i)
        
        if verbose:
            print("episode: {}, epsilon: {}, total reward: {}".format(i, epsilon, total))
        
    return final


env = Minipong(level=1, size=5)

# Number of states
n_state = 1
# Number of actions
n_action = 3
# Number of episodes
episodes = 750
# Number of hidden nodes in the DQN
n_hidden = 150
# Learning rate
lr = 0.001

start = timeit.default_timer()
dqn = DQN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, dqn, episodes, gamma=.9, epsilon=1)

print('\nFinished Training...\n')
stop = timeit.default_timer()
timeTaken = stop - start
print('Time to run the training: ', timeTaken)
print('\n')

"""
After you decide the training to be completed, run 50 test episodes using your trained policy, but with 
epsilon = 0.0 for all 50 episodes. Again, reset the environment at the beginning of each episode. 
Calculate the average over sum-of-rewards-perepisode (call this the Test-Average), 
and the standard deviation (the Test-StandardDeviation). These values indicate how your 
trained agent performs.
"""

# Number of episodes for testing
test_episodes = 50

def simulate_test(env, model):
    final = []
    for i in range(test_episodes):
        state = env.reset()
        done = False
        total = 0
        
        while not done: 
            """
            As epsilon is zero, it always will go to else part which is to predict from model and 
            get the max value as an action
            """
            q_values = model.predict(state)
            action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)
            total += reward
            
            state = next_state
        
        final.append(total)
        print("episode: {}, total reward: {}".format(i, total))
    
    return final

#test_reward = simulate_test(env = env, model = dqn)
#plot_res(test_reward, "Testing", i = 50)

# Test-Average
#test_average = sum(test_reward)/test_episodes

# Test-StandardDeviation
#test_std = np.std(test_reward)

#print('Average training reward: ', test_average)
#print('Test-StandardDeviation is: ', test_std)

