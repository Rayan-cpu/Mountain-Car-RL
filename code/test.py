import analyse

r_factor = [1.0, 10.0]
analyse.heuristic_comparison( r_factor )

'''
def auxiliar_r( self, batch, n, frac ):
    
    #Return the auxiliary reward for the Q-network : the reward grows as x^n as the agent gets closer to the goal. It is equal to 100 (the reward for reaching the goal) when the agent is at the goal.
    #batch : mini-batch over which the reward is computed
    #n : power of the polynomial
    #frac : fraction of the goal reward that is given to the agent
    
    max_reward = 100
    x_reward = 0.5
    x_start = -0.5
    a = max_reward / ( x_reward - x_start ) ** n
    is_on_right = batch[:,0] > x_start
    return frac * a * ( (batch[:,0]-x_start) ** n ) * is_on_right + torch.logical_not(is_on_right) * 0. # if the agent is on the left, the reward is 0

with h5py.File( out_file + '.hdf5', 'w' ) as file:
    file.create_dataset( 'dic1', data = data1 )
    file.create_dataset( 'dic2', data = data2 )

'''

# %%
