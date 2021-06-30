    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:46:12 2020

@author: oliver
"""
import numpy as np
from minipong import Minipong
from matplotlib import pyplot as plt

size = 5
pong = Minipong(level=1, size=size)

# create complete dataset
rows = pong.xymax * pong.xymax * size
columns = pong.size * pong.size

allpix = np.zeros((rows, columns), dtype=int)
alllabels = np.zeros((rows, 3), dtype=int)
j = 0
for x in range(1, pong.xymax + 1):
    for y in range(1, pong.xymax + 1):
        for p in range(pong.zmax + 1):
            allpix[j, :] = pong.to_pix((x, y, p, 0, 0)).flatten(order='C')
            alllabels[j, :] = [x, y, p]
            j += 1

# save all pictures and all labels in random order, two separate files            
np.random.seed(10)
neworder = np.random.choice(rows, size=rows, replace=False)
allpix = allpix[neworder,:]
alllabels = alllabels[neworder, :]
np.savetxt('allpix.csv', allpix, delimiter=',', fmt='%d')
np.savetxt('alllabels.csv', alllabels, delimiter=',', fmt='%d')    

for i in range(4):
    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(allpix[10+i,:].reshape(15,15), origin='lower' , cmap=plt.cm.coolwarm)
    plt.savefig(f'proto{i+1}.png', dpi=300)
    plt.show()


# number of training and test samples
ntrain = round(0.8 * rows)
ntest = rows - ntrain

trainingpix = allpix[0:ntrain,:]
traininglabels = alllabels[0:ntrain,:]
testingpix = allpix[ntrain:,:]
testinglabels = alllabels[ntrain:,:]

np.savetxt('trainingpix.csv', trainingpix, delimiter=',', fmt='%d')
np.savetxt('traininglabels.csv', traininglabels, delimiter=',', fmt='%d')    
np.savetxt('testingpix.csv', testingpix, delimiter=',', fmt='%d')
np.savetxt('testinglabels.csv', testinglabels, delimiter=',', fmt='%d')    
