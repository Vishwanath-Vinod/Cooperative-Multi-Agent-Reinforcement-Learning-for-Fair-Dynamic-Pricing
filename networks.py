import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MIN_PRICE = 100
MAX_PRICE = 500

class GlobalCritic(nn.Module):

    def __init__(self, n_agent, dim_observation, dim_action,hidden_dim1 = 1024,hidden_dim2 = 512):
        super(GlobalCritic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.fc1 = nn.Linear(obs_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1+act_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        self.init_weights()
        
    def init_weights(self):    
        # Proper Weight Initialization using Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # Optional: Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        
    
    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        x = F.relu(self.fc1(obs))
        combined = T.cat([x, acts], 1)
        x = F.relu(self.fc2(combined))
        result = self.fc3(x)
        return result
    
class LocalCritic(nn.Module):

    def __init__(self,dim_observation,dim_action,hidden_dim1=512,hidden_dim2=256,hidden_dim3 = 128):
        super(LocalCritic,self).__init__()
        self.fc1 = nn.Linear(dim_observation, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1+dim_action, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2,hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3,1)
        self.init_weights()
        
    def init_weights(self):    
        # Proper Weight Initialization using Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        # Optional: Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        
    def forward(self, obs, acts):
        result = F.relu(self.fc1(obs))
        combined = T.cat([result, acts], 1)
        result = F.relu(self.fc2(combined))
        return self.fc4(F.relu(self.fc3(result)))
    
class Actor(nn.Module):
    
    def __init__(self, dim_observation, dim_action,hidden_dim1 = 300,hidden_dim2 = 600):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(dim_observation, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, dim_action)
        self.init_weights()
        
    def init_weights(self): 
        # Proper Weight Initialization using Xavier (Glorot) initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # Optional: Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    # action output between 100 and 500
    def forward(self, obs):
        x = T.relu(self.fc1(obs))
        x = T.relu(self.fc2(x))
        action = (T.tanh(self.fc3(x))*(MAX_PRICE-MIN_PRICE)/2)+(MAX_PRICE+MIN_PRICE)/2
        return action




    

   
