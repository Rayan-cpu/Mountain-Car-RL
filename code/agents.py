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

    @abstractmethod
    def run_episode( self, env ):
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

    def run_episode( self, env ):
        state, info = env.reset()
        done = False
        count = 0
        
        while not done:
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.observe(state, action, next_state, reward)
            state = next_state
            
            done = terminated or truncated
            count += 1
        return count


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
    def __init__(self, epsilon=0.9, gamma=0.99, buffer_len=10000, batch_size=64, optimizer='adam', heuristic=False):
        self.actions = self.init_actions()
        self.eps_start = epsilon # will then decay exponentially to reach 0.05
        self.eps_end = 0.05 # asymptotic value for epsilon
        self.eps_tau = 10000 # characteristic time for the decay
        self.gamma = gamma
        self.replay_buffer = torch.zeros( [buffer_len, 6] ) # (x,v, action, reward, x',v')
        self.buffer_len = buffer_len
        self.iter = 0
        self.ep_loss = 0. # loss for the current episode
        self.ep_reward = 0. # reward for the current episode
        self.batch_size = batch_size
        self.heuristic = heuristic
        self.Qs = MLP( 2, len(self.actions) )
        self.target_Qs = MLP( 2, len(self.actions) )
        self.target_Qs.eval()
        self.target_update_freq = 1000 # frequency to update target (in GD steps)
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.Qs.parameters(), lr=1e-3)
        else:
            self.optimizer = torch.optim.SGD(self.Qs.parameters(), lr=1e-3)

    def observe(self, state, action, next_state, reward):
        '''
        Add a sample to the replay buffer. If the buffer is full, replace the oldest sample.
        state : current state of the environment
        action : action taken by the agent
        next_state : state of the environment after taking the action
        reward : reward received after taking the action
        '''
        if self.heuristic: 
            n = 1
            frac = 1.0e-1
            #print( self.auxiliar_r( batch, n, frac )/reward )
            reward += self.float_auxiliar_r( state[0], n, frac )

        # add to replay buffer 
        sample = torch.cat( [torch.tensor(state), torch.tensor([action]), torch.tensor([reward]) , torch.tensor(next_state)] )

        if self.iter < self.buffer_len:
            self.replay_buffer[self.iter,:] = sample
            self.iter += 1
            return
        
        self.replay_buffer = torch.roll( self.replay_buffer, -1, dims=0 ) # replace oldest sample 
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

    def float_auxiliar_r( self, x, n, frac ):
        '''
        Return the auxiliary reward for the Q-network : the reward grows as x^n as the agent gets closer to the goal. It is equal to 100 (the reward for reaching the goal) when the agent is at the goal.
        batch : mini-batch over which the reward is computed
        n : power of the polynomial
        frac : fraction of the goal reward that is given to the agent
        '''
        max_reward = 100
        x_reward = 0.5
        x_start = -0.5
        a = max_reward / ( x_reward - x_start ) ** n
        is_on_right = x > x_start
        return frac * a * ( (x-x_start) ** n ) * is_on_right + (1-is_on_right) * 0. # if the agent is on the left, the reward is 0
    
    def auxiliar_r( self, batch, n, frac ):
        '''
        Return the auxiliary reward for the Q-network : the reward grows as x^n as the agent gets closer to the goal. It is equal to 100 (the reward for reaching the goal) when the agent is at the goal.
        batch : mini-batch over which the reward is computed
        n : power of the polynomial
        frac : fraction of the goal reward that is given to the agent
        '''
        max_reward = 100
        x_reward = 0.5
        x_start = -0.5
        a = max_reward / ( x_reward - x_start ) ** n
        is_on_right = batch[:,0] > x_start
        return frac * a * ( (batch[:,0]-x_start) ** n ) * is_on_right + torch.logical_not(is_on_right) * 0. # if the agent is on the left, the reward is 0

    def huber_loss( self, x, y ):
        '''
        Huber loss function : it is less sensitive to outliers than the mean squared error. 
        x : tensor of predictions
        y : tensor of targets
        '''
        diff = torch.abs( x - y )
        return torch.mean( torch.where( diff < 1, 0.5 * diff ** 2, diff - 0.5 ) )

    def loss_fn( self, batch, huber=True ) :
        '''
        Loss function for the Q-network : we only include part of it in the gradient descent.
        batch : mini-batch over which gradient is estimated
        '''
        target = batch[:,3] + self.gamma * torch.max( self.target_Qs(batch[:,4:]), dim=1 ).values
        actions = batch[:,2].to( torch.int64 ) # to 
        if huber:
            return self.huber_loss( self.Qs(batch[:,0:2])[range(self.batch_size),actions], target )
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
        self.ep_reward += torch.mean( batch[:,3] ).item()

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if done:
            temp = self.ep_loss
            ttemp = self.ep_reward
            self.ep_loss = 0. # reset the loss for the next episode
            self.ep_reward = 0.
            return temp, ttemp
        
        if self.iter % self.target_update_freq == 0:
            self.target_Qs.load_state_dict( self.Qs.state_dict() )        
        return
    
    def run_episode( self, env ):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0.
        count = 0
        
        while not done:
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.observe(state, action, next_state, reward)
            if done and self.iter >= self.buffer_len:
                episode_loss, episode_reward = self.update( done=done )
            else : 
                self.update( done=done )

            state = next_state
            count += 1
        return count, episode_reward, episode_loss