import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(3, 24) # 3 for x y z
        self.hidden_layer = nn.Linear(24, 24)
        self.output_layer = nn.Linear(24, 5) # 5 for 5 actions 

        # Q value actions
        def forward(self, x):
            x = torch.relu(self.input_layer(x))
            x = torch.relu(self.hidden_layer(x))
            x = self.output_layer(x)
            return x

class ReplayBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen = max_size) # removes oldest tests when reaches max memory size
    
    def add(self, experience):
        self.memory.append(experience) # adds the experience to the memory queue. 
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) # randomly selects batch_size amount of unique elements from deque - same experience can't be choosen in one call
    
    def __len__(self):
        return len(self.memory) # how many experiences are stored
    
def start_model():
    model = DQN()
    adam_optimizer = optim.Adam(model.parameters(), lr=0.001) # don't ask me how this works. idfk man. (0.001 learning rate from adam's optimizer)
    return model, adam_optimizer