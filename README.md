# Project 1: Navigation
Deep Reinforcement Learning Nanodegree - Udacity

Iván Vallés Pérez - October 2018

## Introduction

This project shows how to train a DQN algorithm over a Unity environment consisting of a big square world full of yellow and blue bananas. The goal is to collect as much yellow bananas as possible and avoid the blue ones.

![Trained Agent](./img/banana_normal.gif)

## Environment dynamics
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.



## Getting Started

### Python set-up
The following libraries are required
- `numpy`
- `unityagents`
- `pytorch`
- `matplotlib`
- `pandas`

### Environment set-up
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions to run the agent

Follow the instructions in `Report.ipynb` to get started with training your own agent! This code will invoke the following modules of the project
- `agent.py`: contains a class for the agent to orchestrate all the training process. It will be responsible for collecting samples, interacting with the environment, performing training steps and choosing actions.
- `models.py`: contains the model which will be the brain of the agent. It is responsible of implementing the technical part of the neural network using pytorch. It will be used by the agent class as an API. Note, it contains more than one model because the pixels version of the environment is intended to be implemented and appended to this report when it is finished.
- `rl_utilities`: contains different functions for helping the agent overcome the necessary tasks (e.g. experience replay).

