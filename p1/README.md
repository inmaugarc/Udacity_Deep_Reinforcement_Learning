# banana-navigation
Udacity deep reinforcement learning navigation project

</br>

## Project details
In this project I will train an agent to navigate (and collect bananas!) in a large, square world. 

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


</br>

## Getting started
The project is written in a Jupyter Notebook.
To install all the packages, you have to run this command:

```
!pip -q install ./python
```
And then, run all the cells inside the notebook

</br>

## Instructions

The project consists of the following files:

* Navigation.ipynb - the main Jupyter Notebook file
* my_dqn_agent.py - the Agent class
* my_model.py - the DQN model
* checkpoint.pth, checkpoint1.pth, checkpoint2.pth - saved trained models to use
* Report.pdf - a description of the implementation

To select a Dueling DQN model, you have to set the Dueling variable to True. In other case, set this variable to False.

To select a Prioritize Experience Replay, you have to set the corresponding variable to True.In other case, set this variable to False.

```
agent = Agent(state_size=37, action_size=4, seed=0, dueling=False, prioritize=False)
```


