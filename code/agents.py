import random
import torch 
import numpy as np
from abc import ABC, abstractmethod
from utility import MLP, ReplayBuffer
torch.autograd.set_detect_anomaly(True)

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


class DQNAgent(Agent) :
    def __init__(self, epsilon=0.9, gamma=0.99, buffer_len=10000, batch_size=64, optimizer='adam', heuristic=False, pre_train_steps=0):
        self.actions = self.init_actions()

        self.eps_start = epsilon # will then decay exponentially to reach 0.05
        self.eps_end = 0.05 # asymptotic value for epsilon
        self.eps_tau = 20000 # characteristic time for the decay

        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_len)
        self.batch_size = batch_size
        self.iter = 0

        self.ep_loss = 0. # loss for the current episode
        self.ep_env_reward = 0. # reward for the current episode 
        self.ep_aux_reward = 0. # reward for the current episode 
        self.heuristic = heuristic

        self.Qs = MLP( 2, len(self.actions) )
        self.target_Qs = MLP( 2, len(self.actions) )
        self.optimizer = torch.optim.Adam(self.Qs.parameters(), lr=1e-3)
        self.target_Qs.eval()
        self.target_Qs_update_freq = 2000 # frequency to update target (in GD steps)
        self.pre_train_steps = pre_train_steps # number of batches on which to 
        # train the RND network before starting the DQN training
    
    @abstractmethod
    def float_auxiliar_r(self, state, next_state):
        '''
        Return the auxiliary reward associated to the given state.
        '''
        pass

    def observe(self, state, action, next_state, reward):
        '''
        Add a sample to the replay buffer. If the buffer is full, replace the oldest sample.
        state : current state of the environment
        action : action taken by the agent
        next_state : state of the environment after taking the action
        reward : reward received after taking the action
        '''
        self.ep_env_reward += reward
        aux_reward = self.float_auxiliar_r( state, next_state )
        self.ep_aux_reward += aux_reward

        reward += aux_reward
        self.buffer.update( state, action, next_state, reward )
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
            action_index = torch.argmax( self.Qs(torch.tensor(state)) )
            return self.actions[action_index]

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
    
    @abstractmethod
    def sub_update(self, batch):
        '''
        Function to be called within the update method. It is used to train the RND network without having to put if conditions in the update method etc.
        '''
        pass

    def update( self, done ):
        '''
        Train the model using a mini-batch from the replay buffer
        ''' 
        if self.iter < self.buffer.len: # first collect enough samples
            return
        if self.iter % self.target_Qs_update_freq == 0:
            self.target_Qs.load_state_dict( self.Qs.state_dict() )

        batch = self.buffer.new_batch( self.batch_size )
        self.sub_update( batch )

        # train the Q-network once the RND network is good enough
        if self.iter > self.buffer.len + self.pre_train_steps:
            #batch_ = self.buffer.new_batch( self.batch_size )
            self.Qs.train()
            loss = self.loss_fn( batch )
            self.ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            #self.ep_env_reward += torch.mean( batch[:,3] ).item()

        if done:
            ep_loss_ = self.ep_loss
            ep_env_reward_ = self.ep_env_reward
            ep_aux_reward_ = self.ep_aux_reward
            self.ep_loss = 0. # reset the loss for the next episode
            self.ep_env_reward = 0.
            self.ep_aux_reward = 0.
            return ep_loss_, ep_env_reward_, ep_aux_reward_
                
        return
    
    def run_episode( self, env ):
        state, info = env.reset()
        done = False
        ep_env_reward = 0
        ep_aux_reward = 0
        ep_loss = 0.
        count = 0
        
        while not done:
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.observe(state, action, next_state, reward)
            if done and self.iter > self.buffer.len:
                ep_loss, ep_env_reward, ep_aux_reward = self.update( done=done )
                #ep_aux_reward += 100. # reward for reaching the goal, not added by 
            else : 
                self.update( done=done )

            state = next_state
            count += 1
        return count, ep_env_reward/count, ep_aux_reward/count, ep_loss # duration, normalised cumulated reward, loss

class DQNAgentHeuristic(DQNAgent):
    def __init__(self, degree=3, frac=1.0-1, epsilon=0.9, gamma=0.99, buffer_len=10000, batch_size=64, pre_train_steps=0):
        super().__init__(epsilon, gamma, buffer_len, batch_size, pre_train_steps)
        self.frac = frac
        self.degree = degree

    def sub_update(self, batch):
        '''
        This does as heuristics are not trained.
        '''
        pass

    def float_auxiliar_r(self, state, next_state):
        '''
        Return the auxiliary reward for the Q-network : the reward grows as x^n as the agent gets closer to the goal. It is equal to 100 (the reward for reaching the goal) when the agent is at the goal.
        batch : mini-batch over which the reward is computed
        n : power of the polynomial
        frac : fraction of the goal reward that is given to the agent
        '''
        x = state[0]
        max_reward = 100
        x_reward = 0.5
        x_start = -0.5
        a = max_reward / ( x_reward - x_start ) ** self.degree
        is_on_right = x > x_start
        return self.frac * a * ( (x-x_start) ** self.degree ) * is_on_right + (1-is_on_right) * 0. # if the agent is on the left, the reward is 0

class DQNAgentRND(DQNAgent) :
    def __init__(self, reward_factor=1.0, epsilon=0.9, gamma=0.99, buffer_len=10000, batch_size=64, pre_train_steps=1000):
        super().__init__(epsilon, gamma, buffer_len, batch_size, pre_train_steps=pre_train_steps)
        self.RND = MLP( 2, 1 )
        self.RND_optimizer = torch.optim.Adam(self.RND.parameters(), lr=1e-3)
        self.RND_target = MLP( 2, 1 )
        self.RND_target.eval()
        self.reward_mean = 0.
        self.reward_var = 1.
        self.reward_factor = reward_factor
        self.train_index = 0
        
    def sub_update(self, batch):
        '''
        Train the RND network. If the pre-training is done, we compute the auxiliary rewards of the 
        buffer and use them to initialise the reward mean and variance. We then also add the corresponding
        auxiliary rewards to the buffer (this makes sense right ??).
        '''
        self.RND.train()
        RND_loss = torch.mean( ( self.RND_target(batch[:,4:]) - self.RND(batch[:,4:]) ) **2 )
        print('foo')
        RND_loss.backward()
        self.RND_optimizer.step()
        self.RND_optimizer.zero_grad()
        self.train_index += 1
        # the error I have is usually due to calling backward() multiple times on the same graph
        # here can't find the second call to backward() as it is not the one associated to the Q-network (tested it, we don't enter the if)

        print(self.train_index)
        if self.train_index == self.pre_train_steps:
            self.RND.eval()
            buffer_values = self.buffer.values
            buffer_reward = ( self.RND(buffer_values[:,4:]) - \
                        self.RND_target(buffer_values[:,4:]) ) ** 2
            #buffer_reward = ( self.RND(self.buffer.values[:,4:]) - \
            #            self.RND_target(self.buffer.values[:,4:]) ) ** 2
            self.reward_mean = torch.mean( buffer_reward )
            self.reward_var = torch.var( buffer_reward ) 
            norm_reward = ( buffer_reward - self.reward_mean ) / torch.sqrt(self.reward_var)
            # clamp the reward to avoid numerical issues : between -5 and 5
            print('hey')
            temp = buffer_values[:,3] + self.reward_factor * torch.clamp( norm_reward, -5, 5 ).reshape(-1)
            
            self.buffer.values[:,3] = temp.clone() # this modifies the batch, is this problematic ??
            #self.buffer.values[:,3] += self.reward_factor * torch.clamp( norm_reward, -5, 5 ).reshape(-1)




    def float_auxiliar_r(self, state, next_state):
        if self.train_index < self.pre_train_steps:
            return 0.
        next_state_tensor = torch.tensor(next_state)
        self.RND.eval()
        reward = ( self.RND(next_state_tensor) - self.RND_target(next_state_tensor) ) ** 2
        self.reward_mean = (self.reward_mean * self.train_index + reward) / (self.train_index + 1)
        self.reward_var = (self.reward_var * self.train_index + (reward - self.reward_mean) ** 2) / (self.train_index + 1)
        norm_reward = ( reward - self.reward_mean ) / torch.sqrt(self.reward_var)
        return self.reward_factor * torch.clamp( norm_reward, -5, 5 )



        
