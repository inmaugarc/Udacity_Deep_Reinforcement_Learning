# Udacity_Deep_Reinforcement_Learning
Code from Udacity DRL master course
# Project 3 Collaboration & Competition

## Getting Started

This is the implementation of the third project of the Course "Deep Reinforcement Learning" of Udacity. 

### Goal
In this project, I build a reinforcement learning (RL) that will work with the Tennis environment. 
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
 
#### Prerequisites

The code is written in PyTorch and Python 3.6. I trained the agents using the Udacity workspace with GPU enabled. To run the code in this repository on a personal computer, follow the instructions below:


1. Create and activate a new environment with Python 3.6

Linux or Mac:

conda create --name drlnd python=3.6

source activate drlnd
Windows:

conda create --name drlnd python=3.6

activate drlnd

2. Install of OpenAI gym in the environment


    (drlnd) $ git clone https://github.com/openai/gym.git
    (drlnd) $ cd gym
    (drlnd) $ pip install -e .


3. After installing gym, you must install Udacity's requirement



    (drlnd) $ git clone https://github.com/udacity/deep-reinforcement-learning.git
    (drlnd) $ cd deep-reinforcement-learning/python
    (drlnd) $ pip install .

Then you create a Jupiter notebook kernel that can run the Unity environment provided by Udacity as follows:

(drlnd) $ python -m ipykernel install --user --name droned --display-name "drlnd"

4. Install all the code and aditional dependencies


a. Download the Unity environment 
    Linux: [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    Mac OSX: [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    Windows (32-bit): [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    Windows (64-bit): [click here] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)



(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

b. Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.


5. Instructions
	You can run the code using the following command in your terminal 
	
	jupyter-notebook Tennis.ipynb

Be sure that in the same folder there are agent.py, model.py and replay_buffer.py that contains the modules for the agent (actor and critic) and the replay buffer.



6. The training can be run directly in the notebook

Training the Agents

Tennis.ipynb is a Jupyter notebook that can be used to train the multi-agent DDPG model

The Trained Actor and Critic Networks

- checkpoint_actor1.pth contains the weights of the best actor 1 network 
- checkpoint_critic1.pth contains the weights of the best critic 1 network 
- checkpoint_actor2.pth contains the weights of the best actor 2 network 
- checkpoint_critic2.pth contains the weights of the best critic 2 network 

If you would like to observe how the agent performs, you have to be sure that the enviroment is set to False for training and you could load the pre-calculated weights for both agents: actor and critic.

7. Code Description

The environment described above was solved using an actor-critic algorithm, specifically the multi-agent deep deterministic policy gradient (DDPG) method. All of the files below can be found in the code folder.
Module Descriptions

- model.py defines the actor and critic networks
- agent.py defines the MDDPG agents
- replay_buffer.py defines the Replay Buffer 




 


 