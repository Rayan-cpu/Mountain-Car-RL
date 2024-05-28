import random
import torch 
import numpy as np
from abc import ABC, abstractmethod
from utility import MLP, ReplayBuffer
#torch.autograd.set_detect_anomaly(True) # this is to get more info about 
# problem with autograd ( it 10x the computation time tho :/ )

class Agent(ABC):
    # Abstract base class for all agents, defines the mandatory methods.
    def __init__(self, eval_mode):
        self.eval_mode = eval_mode
        pass

    @abstractmethod
    def observe(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def select_action(self, state):
        # no access to the environment as there will be no illegal actions
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def run_episode( self, env ) -> dict:
        pass

    @abstractmethod
    def save_training(self, filename):
        # save the model to a file
        pass

    actions = [0,1,2] # we can move left, stay or move right (same across all agents)
    full_ep_len = 200

class RandomAgent(Agent):
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state): 
        return random.choice( self.actions )

    def update(self):
        pass

    def run_episode( self, env ):
        state, info = env.reset()
        done = False
        results = {'duration' : 0}
        
        while not done:
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            self.observe(state, action, next_state, reward)
            state = next_state
            
            done = terminated or truncated
            results['duration'] += 1
        return results
    
    def save_training(self, filename): # nothing to save
        pass 

class DQNAgent(Agent) :
    def __init__(self, epsilon=0.9, gamma=0.99, buffer_len=50, batch_size=64, pre_train_steps=0, update_period=1, load_from=None):
        self.eps_start = epsilon # will then decay exponentially to eps_end
        self.eps_end = 0.05 # asymptotic value for epsilon
        self.eps_tau = 100*self.full_ep_len # characteristic time for the decay

        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_len*self.full_ep_len) # buffel_len in full episodes
        self.batch_size = batch_size
        self.iter = 0

        self.ep_loss = 0. # loss for the current episode
        self.ep_env_reward = 0. # reward for the current episode 
        self.ep_aux_reward = 0. # reward for the current episode 

        self.Qs = MLP( 2, len(self.actions) )
        self.target_Qs = MLP( 2, len(self.actions) )
        self.optimizer = torch.optim.Adam(self.Qs.parameters(), lr=1e-3)
        self.target_Qs.eval()
        self.target_Qs_update_freq = 2000 # frequency to update target (in GD steps)
        self.pre_train_steps = pre_train_steps # number of GD steps on which to 
        # train the auxiliary reward before starting the DQN training
        self.update_period = update_period
        self.char_state = []
        self.got_char_low = False
        self.got_char_high = False

        eval_mode = False
        if load_from is not None:
            self.Qs.load_state_dict( torch.load(f'{load_from}.pt') )
            eval_mode = True
        super().__init__(eval_mode)
    
    @abstractmethod
    def float_auxiliar_r(self, state, next_state):
        '''
        Return the auxiliary reward associated to the given state. in the general function no auxilary reward is present 
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
        self.ep_env_reward += reward # add the environmental reward
        aux_reward = self.float_auxiliar_r( state, next_state )
        self.ep_aux_reward += aux_reward

        reward += aux_reward # add the auxilary reward 
        self.buffer.update( state, action, next_state, reward )
        
        return

    def select_action(self, state): 
        '''
        Select an action using an epsilon-greedy policy. 
        state : current state of the environment
        '''
        self.Qs.eval()
        Qs = self.Qs(torch.tensor(state))
        action_index = torch.argmax( Qs )

        if self.eval_mode:
            return self.actions[action_index]            

        # otherwise : epsilon greedy policy
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp( -self.iter/self.eps_tau )
        if random.random() < self.epsilon:
            return random.choice( self.actions )
        else:
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
        actions = batch[:,2].to( torch.int64 ) # to use as index 
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
        temp = self.buffer.len + self.pre_train_steps + 1
        if self.iter > temp and self.iter % self.update_period == 0:
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
    
    def save_char_states( self, duration ):
        '''
        Save the characteristic solutions of the environment to a file.
        '''
        if not self.got_char_high :
            if duration > 150 and duration < 170:
                self.got_char_high = True
                self.char_state = np.array(self.char_state)
                np.save('char_state_high.npy', self.char_state)
        if not self.got_char_low :
            if duration > 90 and duration < 120:
                self.got_char_low = True
                self.char_state = np.array(self.char_state)
                np.save('char_state_low.npy', self.char_state)
        self.char_state = []
        return
    
    def run_episode( self, env ):
        state, info = env.reset()
        done = False
        results = {'duration' : 0}
        if not self.eval_mode: # otherwise only need duration
            results['ep_env_reward'] = 0.
            results['ep_aux_reward'] = 0.
            results['ep_loss'] = 0.
        
        while not done:
            action = self.select_action(state) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.char_state.append( next_state )
            if not self.eval_mode:
                self.observe(state, action, next_state, reward)
                if done and self.iter > self.buffer.len:
                    results['ep_loss'], results['ep_env_reward'], results['ep_aux_reward'] = self.update( done=done )
                else : 
                    self.update( done=done )
            
            state = next_state
            results['duration'] += 1
        
        if not (self.got_char_low and self.got_char_high):
            self.save_char_states(results['duration'])

        return results

    
    def save_training(self, filename): 
        '''
        Save the model to a file, so that it can be run in evaluation mode later.
        '''
        torch.save(self.Qs.state_dict(), f'{filename}.pt')
        return 
    

class DQNVanilla(DQNAgent):
    def __init__(self, epsilon=0.9, gamma=0.99, buffer_len=50, batch_size=64, pre_train_steps=0, update_period=1, load_from=None):
        super().__init__(epsilon, gamma, buffer_len, batch_size, pre_train_steps, update_period, load_from)

    def sub_update(self, batch):
        pass

    def float_auxiliar_r(self, state, next_state):
        return 0.

class DQNAgentHeuristic(DQNAgent):
    def __init__(self, degree=3, frac=1.0-1, epsilon=0.9, gamma=0.99, buffer_len=50, batch_size=64, pre_train_steps=0, update_period=1, load_from=None):
        super().__init__(epsilon, gamma, buffer_len, batch_size, pre_train_steps, update_period, load_from)
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
        x_reward = 0.5
        x_start = -0.5

        #max_reward = 100
        a = self.frac / ( ( x_reward - x_start ) ** self.degree )
        is_on_right = x > x_start
        return is_on_right * ( a * (x-x_start) ** self.degree ) - self.frac # if the agent is on the left, the reward is 0

class DQNAgentRND(DQNAgent) :
    def __init__(self, reward_factor=1.0, epsilon=0.9, gamma=0.99, buffer_len=50, batch_size=64, pre_train_steps=100, update_period=1, load_from=None):
        super().__init__(epsilon, gamma, buffer_len, batch_size, pre_train_steps=pre_train_steps, update_period=update_period) # have to add load_from
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
        if self.train_index == self.pre_train_steps:
            self.RND.eval()
            buffer_reward = ( self.RND(self.buffer.values[:,4:]) - \
                        self.RND_target(self.buffer.values[:,4:]) ) ** 2
            self.reward_mean = np.mean( buffer_reward.detach().numpy() )
            self.reward_var = np.var( buffer_reward.detach().numpy() ) 
            
        # check if this makes sense ???
        if self.train_index % self.update_period == 0 : 
            self.RND.train()
            RND_loss = torch.mean( ( self.RND_target(batch[:,4:]) - self.RND(batch[:,4:]) ) **2 )
            RND_loss.backward()
            self.RND_optimizer.step()
            self.RND_optimizer.zero_grad()
            self.train_index += 1

    def float_auxiliar_r(self, state, next_state):
        if self.train_index < self.pre_train_steps:
            return 0.
        next_state_tensor = torch.tensor(next_state)
        self.RND.eval()
        reward = ( self.RND(next_state_tensor) - self.RND_target(next_state_tensor) ) ** 2
        norm_reward = ( reward - self.reward_mean ) / np.sqrt(self.reward_var)
        self.reward_mean = (self.reward_mean * self.train_index + reward.item()) / (self.train_index + 1)
        self.reward_var = (self.reward_var * self.train_index + (reward.item() - self.reward_mean) ** 2) \
                            / (self.train_index + 1)
        return self.reward_factor * torch.clamp( norm_reward, -5, 5 )


class DynaAgent(Agent):

    def __init__(self,k = 3, x_step=0.025,v_step=0.005, gamma=0.99, epsilon=0.9):
        self.discr_step = np.array([x_step,v_step])
        self.gamma = gamma
        self.epsilon = epsilon
        self.k = k

        self.discrete_positions_attribute, self.discrete_velocities_attribute = self.discrete_state_vectors()
        self.number_of_states = len(self.discrete_positions_attribute)*len(self.discrete_velocities_attribute)
        self.P_tensor = (1/self.number_of_states)*np.ones( ( self.number_of_states , 3 , self.number_of_states ) ) # P(s,a,s)
        self.R_matrix = np.zeros( ( self.number_of_states , 3 ) )
        self.Q_matrix = np.zeros( ( self.number_of_states , 3 ) )

        # The matrix Count_matrix (index s,a,s') will count the number of times i have done : s -> a -> s'
        self.Count_matrix = np.zeros( ( self.number_of_states , 3 )) 

        # The schedule of the epsilon-greedy : 
        self.eps_start = epsilon # will then decay exponentially to eps_end
        self.eps_end = 0.05 # asymptotic value for epsilon
        self.eps_tau = 100*self.full_ep_len # characteristic time for the decay
        self.iter = 0
        self.step_index = 0

        # current state, action, next state and reward : 
        self.index_current_state = 0
        self.index_current_action = 0
        self.index_current_next_state = 0 
        self.current_reward = 0

        # characterstic trajectories (initialization here does not matter, the only purpose is to initialize 
        # to something non empty in order for the code to run even though no-characterstic state has been selected): 
        self.characteristic_trajectory_1 = [[10,10],[10,10]] # initialize outside of the interesting box !
        self.characteristic_trajectory_2 = [[10,10],[10,10]]
        self.characteristic_trajectory_3 = [[10,10],[10,10]]
        self.characteristic_trajectory_4 = [[10,10],[10,10]]

        # boolean variables that indicate if the characteristic Q matrices have already been reported (we only want one of each):
        self.Q1_filled = False
        self.Q2_filled = False
        self.Q3_filled = False
        self.characteristic_Q_1 = np.zeros( ( self.number_of_states , 3 ) )
        self.characteristic_Q_2 = np.zeros( ( self.number_of_states , 3 ) )
        self.characteristic_Q_3 = np.zeros( ( self.number_of_states , 3 ) )
        # boolean variables that indicate if the characteristic Count matrices have already been reported (we only want one of each):
        self.Count1_filled = False
        self.Count2_filled = False
        self.Count3_filled = False
        self.characteristic_Count_1 = np.zeros( ( self.number_of_states , 3 ) )
        self.characteristic_Count_2 = np.zeros( ( self.number_of_states , 3 ) )
        self.characteristic_Count_3 = np.zeros( ( self.number_of_states , 3 ) )
        self.episode_number = 0
        # final tables to print :
        self.Q_changes_list = []
        self.Rs_list = []

    def generate_points(self, lim_left, lim_right, step_size): # CONFIRM 
        num_steps = int((lim_right - lim_left) // step_size) # Calculate the number of steps needed to include lim_right
        points = [lim_right - i * step_size for i in range(num_steps + 1)] # Generate the grid points starting from lim_right and moving left
        points.reverse() # Reverse the list to start from the leftmost point
        return points
    

    def discrete_state_vectors(self): # CONFIRMED
        # discover the discrete states (as a matrix first and put them in a 1D vector)
        left_pos = -1.2
        right_pos = 0.6
        left_vit = -0.07
        right_vit = 0.07
        possible_positions = self.generate_points(left_pos,right_pos,self.discr_step[0])
        possible_velocities = self.generate_points(left_vit,right_vit,self.discr_step[1])

        return possible_positions,possible_velocities

    def find_closest(self, array, value): # CONFIRMED 
        '''
        output : index of closest element of array to the value we look at.
        '''
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def index_vec_from_indices(self,index_position,index_velocity,how_many_positions): # CONFIRMED
        final_index = index_position + index_velocity * how_many_positions
        return final_index

    def discretize(self, s): 
        '''
        input : continuous state s.
        output : indices of state s in the vector of descretized states. 
        '''
        # discretizing: associating to the position of the state the index of the closest position
        possible_positions = self.discrete_positions_attribute
        possible_velocities = self.discrete_velocities_attribute

        index_pos_s = self.find_closest(possible_positions,s[0]) 
        index_vit_s = self.find_closest(possible_velocities,s[1]) 
        # now i can place my two states in the discrete grid ! 
        index_s = self.index_vec_from_indices(index_pos_s,index_vit_s,len(possible_positions))
        return index_s 
    
    def observe(self, state, action, next_state, reward):
        '''
        This function updates the probability, rewards and count matrices !
        '''
        index_state = self.discretize(state) # discretize first state
        index_next_state = self.discretize(next_state) # discretize second state 
        
        self.step_index = 0

        hot_encoded_vector = np.zeros(self.number_of_states)
        hot_encoded_vector[index_next_state] = 1

        visit_number = self.Count_matrix[index_state,action] # add 1 in the counter matrix !
        self.Count_matrix[index_state,action] = visit_number + 1

        # update the probability matrix !
        self.P_tensor[index_state,action,:] = (visit_number * self.P_tensor[index_state,action,:] + hot_encoded_vector) / (visit_number+1)

        # update the rewards matrix ! 
        self.R_matrix[index_state,action] = (visit_number * self.R_matrix[index_state,action] + reward) / (visit_number+1)

        self.iter = self.iter + 1 

        self.index_current_state = index_state    
        self.index_current_next_state = index_next_state    
        self.index_current_action = action    
        self.current_reward = reward     

        return 

    def select_action(self, state):
        vector_Q_values = self.Q_matrix[self.discretize(state),:] # the Q-values corresponding to the state (for different actions)
        # epsilon decay
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp( -self.iter/self.eps_tau )
        
        # epsilon greedy policy
        if random.random() < self.epsilon:
            return random.choice( self.actions )
        else:
            vector_Q_values = torch.tensor(vector_Q_values, dtype=torch.float32) 
            action_index = torch.argmax( vector_Q_values )
            return self.actions[action_index]
        
    def Q_update_equation(self,s,a):
        P_s_a = self.P_tensor[s, a, :]  # shape (noOfstates,) / Extracting the slice of the transition probabilities for the given state and action
        max_Q_s_prime = np.max(self.Q_matrix, axis=1)  # shape (noOfstates,) / extracting the max q values !
        sum_P_times_Q = np.sum(P_s_a * max_Q_s_prime)  
        self.Q_matrix[s,a] = self.R_matrix[s,a] + self.gamma * sum_P_times_Q

        #self.Q_matrix = np.copy(self.Q_matrix) # ???? to avoid list with functions ? 
        return 
    
    def non_zero_positions(self,count_matrix): 
        '''
        This function returns the positions of the visited (s,a) pairs
        '''
        non_zero_indices = np.argwhere(count_matrix != 0) # Get the positions where the sum is non-zero
        return non_zero_indices 
    
    def random_select_action(self):
        probabilities = [1/3, 1/3, 1/3]
        action = np.random.choice([0, 1, 2], p=probabilities)
        return action

    def update(self):
        # should update the q value (s,a) !
        # i need to rely on the reward observed
        # update equation : 
        self.Q_update_equation( self.index_current_state , self.index_current_action)
        non_zero_counts = self.non_zero_positions(self.Count_matrix) # select k random state action pairs from the Counts_matrix.
        # Now i will have to take k random numbers and update them ! 
        if self.k != 0:
            if non_zero_counts.shape[0] > self.k :
                #print(self.iter)
                #print(non_zero_counts)
                random_indices_state_action = np.random.choice(non_zero_counts.shape[0], self.k, replace=False) # Get k random row indices (state indices)
                selected_states = non_zero_counts[random_indices_state_action] # the set of random states i selected

                for random_state in selected_states:
                    self.Q_update_equation(random_state[0],random_state[1]) 
        
        return
    
    def Q_values_distance(self,Q_before,Q_after):
        squared_differences = np.square(Q_before - Q_after) 
        norm_difference = np.sqrt(np.sum(squared_differences))
        return norm_difference 
    
    def Q_value_final_format(self,Q_old):
        number_of_pos = len(self.discrete_positions_attribute)
        number_of_vel = len(self.discrete_velocities_attribute)

        Q_formatted = np.zeros((number_of_pos,number_of_vel))
        for i in range(number_of_pos):
            for j in range(number_of_vel):
                Q_formatted[i,j] = np.max(Q_old[i+j*number_of_pos,:])

        return Q_formatted
    
    def Counts_final_format(self,matrix_old):
        number_of_pos = len(self.discrete_positions_attribute)
        number_of_vel = len(self.discrete_velocities_attribute)

        matrix_formatted = np.zeros((number_of_pos,number_of_vel))
        for i in range(number_of_pos):
            for j in range(number_of_vel):
                matrix_formatted[i,j] = np.sum(matrix_old[i+j*number_of_pos,:])

        return matrix_formatted
    
    # Make a matrix that would choose the 4 states to plot 
    # this matrix would check that the duration is good and that the iteration is far enough in the training :
    def store_trajectories(self, duration,table_of_states):
        if 180 < duration < 200 and len(self.characteristic_trajectory_1) == 2 and self.iter > 2500 :
            self.characteristic_trajectory_1[:] = np.copy(table_of_states)
        if 140 < duration < 160 and len(self.characteristic_trajectory_2) == 2 and self.iter > 2500:            
            self.characteristic_trajectory_2[:] = np.copy(table_of_states)
        if 110 < duration < 115 and len(self.characteristic_trajectory_3) == 2 and self.iter > 2500:
            self.characteristic_trajectory_3[:] = np.copy(table_of_states)
        if 80 < duration < 90 and len(self.characteristic_trajectory_4) == 2 and self.iter > 2500:
            self.characteristic_trajectory_4[:] = np.copy(table_of_states)

    def store_Q_matrix(self, duration): # we will change the conditions, cest temporaire
        if 180 < duration < 200 and not self.Q1_filled:# and self.iter > 2500 :
            self.characteristic_Q_1[:] = np.copy(self.Q_matrix)
            self.Q1_filled = True
        if 140 < duration < 160 and not self.Q2_filled:# and self.iter > 2500:     
            self.characteristic_Q_2[:] = np.copy(self.Q_matrix)
            self.Q2_filled = True
        if 110 < duration < 115 and not self.Q3_filled:# and self.iter > 2500:
            self.characteristic_Q_3[:] = np.copy(self.Q_matrix)
            self.Q3_filled = True

    def store_Count_matrix(self, duration): # we will change the conditions, cest temporaire
        if 180 < duration < 200 and not self.Count1_filled:# and self.iter > 2500 :
            self.characteristic_Count_1[:] = np.copy(self.Count_matrix)
            self.Count1_filled = True
        if 140 < duration < 160 and not self.Count2_filled:# and self.iter > 2500:            
            self.characteristic_Count_2[:] = np.copy(self.Count_matrix)
            self.Count2_filled = True
        if 110 < duration < 115 and not self.Count3_filled:# and self.iter > 2500:
            self.characteristic_Count_3[:] = np.copy(self.Count_matrix)
            self.Count3_filled = True
        

    def run_episode( self, env) -> dict:
        state, info = env.reset()
        done = False
        self.episode_number += 1
        results = {'duration' : 0}
        results['ep_env_reward'] = 0
        results['ep_Q_values_change'] = 0
        self.step_index = 0
        total_Q_value_change = 0
        total_reward = 0
        Q_matrix_before = np.copy(self.Q_matrix)
        states_of_episode = []

        while not done:
            action = self.select_action(state) # select an action 
            next_state, reward, terminated, truncated, _ = env.step(action) # make the agent move 
            done = terminated or truncated # check if it has ended
            self.observe(state, action, next_state, reward) # updates R,P, and counts_matrix

            self.update() # update the belief on the Q values 
            self.Rs_list.append(reward)
            total_reward = total_reward + reward
            state = next_state
            results['duration'] += 1

            # conserve the states (that potentially will be injected as a characteristic state if it obeys some criteria) : 
            states_of_episode.append(state)

        #store the characteristic trajectories if relevant (done under conditions put in the functions)
        self.store_trajectories(results['duration'],states_of_episode)
        #store the characteristic Q and count matrices if relevant
        self.store_Q_matrix(results['duration'])
        self.store_Count_matrix(results['duration'])

        results['ep_Q_values_change'] = total_Q_value_change
        results['ep_env_reward'] = total_reward
        Q_matrix_after = np.copy(self.Q_matrix)
        results['ep_Q_values_change'] = self.Q_values_distance(Q_matrix_before,Q_matrix_after)
        return results
    

    def end_episode(self):
        '''
        Returns quantities only needed to be exported once the end of the
        simulation (and not at the end of each episode)
        '''
        Q_matrix_end = self.Q_value_final_format(self.Q_matrix)
        pos_axis_plot = self.discrete_positions_attribute
        vel_axis_plot = self.discrete_velocities_attribute
        Count_matrix = self.Counts_final_format(self.Count_matrix)

        characteristic_trajectory_1 = self.characteristic_trajectory_1
        characteristic_trajectory_2 = self.characteristic_trajectory_2
        characteristic_trajectory_3 = self.characteristic_trajectory_3
        characteristic_trajectory_4 = self.characteristic_trajectory_4

        characteristic_Q_1 = self.Q_value_final_format(self.characteristic_Q_1)
        characteristic_Q_2 = self.Q_value_final_format(self.characteristic_Q_2)
        characteristic_Q_3 = self.Q_value_final_format(self.characteristic_Q_3)

        characteristic_Count_1 = self.Counts_final_format(self.characteristic_Count_1)
        characteristic_Count_2 = self.Counts_final_format(self.characteristic_Count_2)
        characteristic_Count_3 = self.Counts_final_format(self.characteristic_Count_3)
    
        return Q_matrix_end,pos_axis_plot,vel_axis_plot,Count_matrix, characteristic_trajectory_1,characteristic_trajectory_2,characteristic_trajectory_3,characteristic_trajectory_4,characteristic_Q_1,characteristic_Q_2,characteristic_Q_3,characteristic_Count_1,characteristic_Count_2,characteristic_Count_3
    