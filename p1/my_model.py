import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3=256, fc4=256):

        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed   
            the 4 Linear Layers of the NN
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_units = nn.Linear(state_size, fc1_units)
        self.fc2_units = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3)
        self.fc4 = nn.Linear(fc3,fc4)
        self.fc5 = nn.Linear(fc4,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1_units(state))
        x = F.relu(self.fc2_units(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256, fc3=256, fc4=256):

        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed   
            the 4 Linear Layers of the NN
        """
    
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_units = nn.Linear(state_size, fc1_units)
        self.fc2_units = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3)
        self.fc4 = nn.Linear(fc3,fc4)
        
        self.fc5 = nn.Linear(fc4,128)
        self.fc6 = nn.Linear(128,action_size)
        
        self.fc7 = nn.Linear(fc4,128)
        self.fc8 = nn.Linear(128,1)
        
        
    def forward(self,state):
        x = F.relu(self.fc1_units(state))
        x = F.relu(self.fc2_units(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
    
        
        #State values
        value = self.fc8(F.relu(self.fc7(x)))
        
        #Advantage values
        advantage = self.fc6(F.relu(self.fc5(x)))
        
        
        #calculate dueled_value
        dueled_value = value + advantage - advantage.mean()
        
        return dueled_value
    