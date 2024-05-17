#%% 
import pandas as pd
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

low_path = 'char_state_low.npy'
high_path = 'char_state_high.npy'
low = np.load(low_path)
high = np.load(high_path)
x_low = low[:,0]
t_low = np.arange(0, len(x_low))
x_high = high[:,0]
delta_size = x_high.size - x_low.size
t_high = np.arange(0, len(x_high))


fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].scatter(x_low, t_low)
ax[0].scatter(x_high[delta_size:], t_high[:-delta_size])
ax[1].scatter(x_high, t_high)
ax[0].set_xlim(-1.2, 0.6)

#%%
import matplotlib.pyplot as plt
import numpy as np

def float_auxiliar_r(degree, frac, x):
    x_reward = 0.5
    x_start = -0.5
    a = frac / ( ( x_reward - x_start ) ** degree )
    is_on_right = x > x_start
    return is_on_right * ( a * (x-x_start) ** degree ) - frac

# %%

x_left = -1.2
x_right = 0.6
x_start = -0.5
n_points = 500
x = np.linspace(x_left, x_right, n_points)
degree = np.arange(1,5,1)
frac = 100.e-0
for d in degree:
    y = float_auxiliar_r(d, frac, x)
    plt.plot(x, y, label=f'degree={d}')
plt.axvline(x=x_start, color='k', linestyle='--')
plt.xlabel('x')
plt.legend()
plt.ylabel('Reward')
plt.show()

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
'''

# %%
