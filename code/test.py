# %%
import yaml

with open('../configs/test.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

print( prime_service )

with open('../configs/w_test.yaml', 'w') as file:
    yaml.dump(prime_service, file)

# %%
import matplotlib.pyplot as plt
import numpy as np

def float_auxiliar_r(degree, frac, x):
    max_reward = 100
    x_reward = 0.5
    x_start = -0.5
    a = max_reward / ( x_reward - x_start ) ** degree
    is_on_right = x > x_start
    # if the agent is on the left, the reward is 0
    return frac * a * ( (x-x_start) ** degree ) * is_on_right + (1-is_on_right) * 0. 

# %%

x_left = -1.2
x_right = 0.6
x_start = -0.5
n_points = 500
x = np.linspace(x_left, x_right, n_points)
degree = np.arange(1,5,1)
frac = 1.e-0
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
