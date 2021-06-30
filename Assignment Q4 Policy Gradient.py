#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:53:25 2020

@author: spviradiya
"""

import numpy as np
from itertools import count
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from minipong import Minipong
from torch.autograd import Variable
from matplotlib import pyplot as plt

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

class Policyminipong(nn.Module):
    """
    implements a policy module for REINFORCE
    """
    def __init__(self, ninputs, noutputs, gamma = 0.95):
        super(Policyminipong, self).__init__()
        # your code here
        self.fc1 = nn.Linear(ninputs, 128)
        self.fc2 = nn.Linear(128, noutputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

        # discount factor
        self.gamma = gamma
        # action & reward buffer
        self.saved_log_probs = []
        self.rewards = []
        
        # smallest useful value
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)
        return F.softmax(scores, dim=1)
    
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
        
    
    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # standardise returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # compute policy losses, using stored log probs and returns
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # run backprop through all that
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # delete stored values 
        del self.rewards[:]
        del self.saved_log_probs[:]

### main

seed = 1
gamma = 0.95
render = False
finalrender = True
log_interval = 50
running_reward = 10

env = Minipong(level=3, size=5)
#env.seed(seed)
torch.manual_seed(seed)

ninputs = 3
noutputs = 3
policy = Policyminipong(ninputs, noutputs, gamma)

starttime = time.time()
final = []

for i_episode in count(1):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning

        # select action from policy
        action = policy.select_action(state)

        # take the action
        state, reward, done = env.step(action)
        reward = float(reward)     # strange things happen if reward is an int
        if render:
            env.render()
            
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            final.append(ep_reward)
            break

    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    # perform backprop
    policy.update()
    
    # log results
    if i_episode % log_interval == 0:
        print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward))
        plot_res(final, i = i_episode)

    # check if we have solved cart pole
    if running_reward > 300:
        secs = time.time() - starttime
        mins = int(secs/60)
        secs = round(secs - mins * 60.0, 1)
        print("Solved in {}min {}s!".format(mins, secs))
            
        print("Running reward is now {:.2f} and the last episode "
              "runs to {} time steps!".format(running_reward, t))

        if finalrender:
            state = env.reset()
            for t in range(1, 10000):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                env.render()
                if done:
                    break
        
        break


# Number of episodes for testing
test_episodes = 50

def simulate_test(env, model):
    final = []
    for i in range(test_episodes):
        state = env.reset()
        done = False
        total = 0
        
        while not done: 
            
            # select action from policy
            action = policy.select_action(state)
            
            # Take action and add reward to total
            next_state, reward, done = env.step(action)
            total += reward
            
            state = next_state
        
        final.append(total)
        print("episode: {}, total reward: {}".format(i, total))
    
    return final



test_reward = simulate_test(env = env, model = policy)
plot_res(test_reward, "Testing", i = 50)

# Test-Average
test_average = sum(test_reward)/test_episodes

# Test-StandardDeviation
test_std = np.std(test_reward)

print('Average training reward: ', test_average)
print('Test-StandardDeviation is: ', test_std)