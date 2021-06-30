# Object-Prediction-Auto-encoder-for-Images-Reinforcement-Agent-for-Minipong-Game-Python
Train a CNN to predict object positions, Train a convolutional autoencoder, Create a RL agent for Minipong

<b>Click on the below image to see the training video of MiniPong on YouTube. (Please note that it will redirect to YouTube)</b>

[![Trained MiniPong](https://img.youtube.com/vi/acnZVil-4f0/maxresdefault.jpg)](https://youtu.be/acnZVil-4f0)

# Minipong

Two objects appear on the field: a + object as “ball”, and a − paddle that can take different spots, but only in the bottom row. Pixels of the two objects are represented with different the values −1 and 1 while background pixels have the value 0. The two markers at the top corners are fixed (−1 and +1, respectively) and appear in every frame.

<b>Preparation</b> Download the Minipong.py and sprites.py python files. The class Minipong.py implements the pong game simulation. Running sprites.py will create datasets of pong screenshots for your first task.

A new pong game can be created like here:

from minipong import Minipong<br/>
pong = Minipong(level=1, size=5)

In this, level sets the information a RL agent gets from the environment, and size sets the size of the game (in number of different paddle positions). Both paddle and + are 3 pixels wide, and cannot leave the field. A game of size 5 is (15 × 15) pixels, and the ball x- and y-coordinates can be values between 1 and 13. The paddle can be in 5 different locations (from 0 to 4).

## Task 1: Train a CNN to predict object positions

The python program sprites.py creates a training and test set of “minipong” scenes, trainingpix.csv (676 samples) and testingpix.csv (169 samples). Each row represents a 15×15 screenshot (flattened in row-major order). Labels appear in sep- arate files, traininglabels.csv and testlabels.csv. They contain 3 labels for each example (x/y/z), the x/y-coordinates for the + marker with values between 1 and 13, and z between 0...4, for the location of the − paddle.

<b>Steps</b>
<ol>
  <li>Create the datasets by running the sprites.py code.</li>
  <li>Create a CNN that predicts the x-coordinate of the + marker.
    <ul>
      <li>You can (but don’t have to) use an architecture similar to what we used for classifying MNIST, but be aware the input dimensions and outputs are different, so you will have to make at least some changes.</li>
      <li>You can normalise/standardise the data if it helps improve the training.</li>
    </ul>
  </li>
  <li>Create a CNN that predicts all three outputs (x/y/z) from each input . • Compute the accuracy on the test data set.</li>
</ol>

## Task 2: Train a convolutional autoencoder

Instead of predicting positions, create a convolutional autoencoder that compresses the pong screenshots to a small number of bytes (the encoder), and transforms them back to original (in the decoder part).

<b>Steps</b>

<ol>
  <li>Create and train an (undercomplete) convolutional autoencoder and train it using the training data set from the first task.</li>
  <li>You can choose the architecture of the network and size of the representation h = f(x). The goal is to learn a representation that is smaller than the original, and still leads to recognisable reconstructions of the original.</li>
  <li>For the encoder you can use the same architecture that you used for the first task, but you cannot use the labels for training. You can also create a completely dif- ferent architecture.</li>
  <li>(No programming): In theory, what would be the absolute minimal size of the hidden layer representation that allows perfect reconstruction of the original im- age?</li>
</ol>

## Task 3: Create a RL agent for Minipong (level 1)

The code in minipong.py provides an environment to create an agent that can be trained with reinforcement learning (a complete description at the end of this sheet). It uses the objects as described above. The following is a description of the environment dynamics:

<ul>
  <li>The + marker moves a diagonal step at each step of the environment. When it hits the paddle or a wall (on the top, left, or right) it reflects.</li>
  <li>The agent can control the paddle (−), by moving it one 3-pixel slot every step. The agent has three actions available: it can choose to do nothing, or it can move it to the left or right. The paddle cannot be moved outside the boundaries.</li>
  <li>The agent will receive a positive reward when the + reflects from the paddle. In this case, the + may also move by 1 or 2 random pixels to the left or right.</li>
  <li>An episode is finished when + reaches the bottom row without reflecting from the paddle.</li>
</ul>

In a level 1 version of the game, the observed state (the information made available to the agent after each step) consists of one number: dz. It is the relative position of the +, relative to the centre of the paddle: a negative number if + is on one side, a positive one on the other.

For this task, you can initialise pong like this:<br/>
  pong = Minipong(level=1, size=5)

or like this:<br/>
  pong = Minipong(level=1, size=5, normalise = False)
  
In the first version, step() returns normalised values of dz (values between −1...1) for the state, while in the second version it returns pixel differences (−13...13).

<b>Steps</b>

<ol>
  <li>Manually create a policy (no RL) that successfully plays pong, just selecting ac- tions based on the state information. The minipong.py code contains a tem- plate that you can use and modify.</li>
  <li>Create a (tabular or deep) TD agent that learns to play pong. For choosing actions with ε-greedy action selection, set ε = 1, initially, and reduce it during your training to a minimum of 0.1.</li>
  <li>Run your training, resetting after every episode. Store the sum of rewards. After or during the training, plot the total sum of rewards per episode. This plot — the Training Reward plot — indicates the extent to which your agent is learning to improve his cumulative reward. It is your decision when to stop training. It is not required to submit a perfectly performing agent, but show how it learns.</li>
  <li>After you decide the training to be completed, run 50 test episodes using your trained policy, but with ε = 0.0 for all 50 episodes. Again, reset the environment at the beginning of each episode. Calculate the average over sum-of-rewards-per- episode (call this the Test-Average), and the standard deviation (the Test-Standard- Deviation). These values indicate how your trained agent performs.</li>
  <li>If you had initialised pong with pong = Minipong(level=2, size=5), the observed state would consist of 2 values: the ball y-coordinate, and the relative +-− position dz from level 1. Will this additional information help or hurt the learning? (No programming required).</li>
</ol>  
