import random
import torch 
import torch.nn as nn

class MLP( nn.Module ):
    '''
    Multi-layer perceptron class. It is a simple feed-forward neural network with 2 hidden layers of size 64 and ReLU activation functions.
    '''
    def __init__( self, in_dim, out_dim ):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward( self, x ):
        logits = self.linear_relu_stack( x )
        return logits

class ReplayBuffer():
    '''
    Class for the replay buffer. It stores the samples in a tensor and provides methods to easily update its content and generate random batches.
    '''
    def __init__(self, buffer_len):
        self.values = torch.zeros( [buffer_len, 6] ) # (x,v, action, reward, x',v')
        self.len = buffer_len
        self.state_mean = torch.tensor([0.0, 0.0])
        self.state_var = torch.tensor([1.0, 1.0])
        self.index = 0
    
    def update(self, state, action, next_state, reward):
        state_tensor = torch.tensor(state)
        sample = torch.cat( [state_tensor, torch.tensor([action]), torch.tensor([reward]) , torch.tensor(next_state)] )

        if self.index < self.len:
            self.values[self.index,:] = sample
            self.state_mean = (self.state_mean * self.index + state_tensor) / (self.index + 1)
            self.state_var = (self.state_var * self.index + (state_tensor - self.state_mean) ** 2) / (self.index + 1)
            self.index += 1
            return
        
        self.state_mean += (state_tensor - self.values[0,0:2]) / self.len
        self.state_var += ( (state_tensor - self.state_mean) ** 2 - (self.values[0,0:2] - self.state_mean) ** 2 ) / self.len
        self.values = torch.roll( self.values, -1, dims=0 ) # replace oldest sample 
        self.values[-1,:] = sample   
        return
    
    def new_batch( self, batch_size ):
        batch_ind = random.sample( range(self.len), batch_size ) 
        batch = self.values[batch_ind,:]
        return batch