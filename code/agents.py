import random
import torch 
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    # Abstract base class for all agents, defines the mandatory methods.
    @abstractmethod
    def observe(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self):
        pass

    def init_actions(self):
        return [0,1,2] # we can move left, stay or move right


class RandomAgent(Agent):
    def __init__(self):
        self.actions = self.init_actions()
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state): 
        # no access to the environment as there will be no illegal actions
        return random.choice( self.actions )

    def update(self):
        pass


# Multi layer perceptron class
class MLP( nn.Module ):
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


class DQNAgent(Agent) :
    def __init__(self, epsilon=0.9, gamma=0.99, buffer_len=10000, batch_size=64, optimizer='adam'):
        self.actions = self.init_actions()
        self.eps_start = epsilon # will then decay exponentially to reach 0.05
        self.eps_end = 0.05 # asymptotic value for epsilon
        self.eps_tau = 1000 # characteristic time for the decay
        self.gamma = gamma
        self.replay_buffer = torch.zeros( [buffer_len, 6] ) # (x,v, action, reward, x',v')
        self.buffer_len = buffer_len
        self.iter = 0
        self.ep_loss = 0. # loss for the current episode
        self.batch_size = batch_size
        self.Qs = MLP( 2, len(self.actions) )
        if optimizer == 'adam':
            self.optimizer = torch.optim.SGD(self.Qs.parameters(), lr=1e-3)
        else:
            self.optimizer = torch.optim.SGD(self.Qs.parameters(), lr=1e-3)
    # Qs is attribute of the agent

    def observe(self, state, action, next_state, reward):
        '''
        Add a sample to the replay buffer. If the buffer is full, replace the oldest sample.
        state : current state of the environment
        action : action taken by the agent
        next_state : state of the environment after taking the action
        reward : reward received after taking the action
        '''
        # add to replay buffer 
        sample = torch.cat( [torch.tensor(state), torch.tensor([action]), torch.tensor([reward]) , torch.tensor(next_state)] )

        if self.iter < self.buffer_len:
            self.replay_buffer[self.iter,:] = sample
            self.iter += 1
            return
        
        self.replay_buffer = torch.roll( self.replay_buffer, -1 ) # replace oldest sample 
        self.replay_buffer[-1,:] = sample                         # with the new one
        self.iter += 1
        return


    def select_action(self, state): 
        '''
        Select an action using an epsilon-greedy policy. 
        state : current state of the environment
        '''
        self.Qs.eval()

        # epsilon decay
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp( -self.iter/self.eps_tau )
        # epsilon greedy policy
        if random.random() < self.epsilon:
            return random.choice( self.actions )
        else:
            # input of NN is tensor
            action_index = torch.argmax( self.Qs(torch.tensor(state)) )
            return self.actions[action_index]

    def loss_fn( self, batch ) :
        '''
        Loss function for the Q-network : we only include part of it in the gradient descent.
        batch : mini-batch over which gradient is estimated
        '''
        with torch.no_grad(): # r + gamma * max_a Q(s',a)
            max = torch.max( self.Qs(batch[:,4:]), dim=1 ) 
            target = batch[:,3] + self.gamma * max.values
        actions = batch[:,2].to( torch.int64 ) # to 
        return torch.mean( ( target - self.Qs(batch[:,0:2])[range(self.batch_size),actions] ) ** 2 )
    
    def update( self, done ):
        '''
        Train the model using a mini-batch from the replay buffer
        '''
        # first collect enough samples 
        if self.iter < self.buffer_len:
            return

        self.Qs.train()
        batch_ind = random.sample( range(self.buffer_len), self.batch_size ) 
        batch = self.replay_buffer[batch_ind,:]
        loss = self.loss_fn( batch )
        self.ep_loss += loss.item() 

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if done:
            temp = self.ep_loss
            self.ep_loss = 0. # reset the loss for the next episode
            return temp
        return
        #if batch % 100 == 0:
        #    loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
