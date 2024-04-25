import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

output_dir = '../output/'
filename = 'dqn_agent'
fig_dir = '../figs/'

# load the data from the .h5 file
with h5.File(f'{output_dir}{filename}.h5', 'r') as f:
    eps = f['eps'][:]
    count = f['count'][:]
    norm_ep_env_reward = f['norm_ep_env_reward'][:]
    norm_ep_aux_reward = f['norm_ep_aux_reward'][:]
    ep_loss = f['ep_loss'][:]

plt.scatter(eps, count)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.savefig( f'{fig_dir}dqn_duration.png')
plt.figure()
plt.scatter(eps, norm_ep_env_reward, label='Environment')
plt.scatter(eps, norm_ep_aux_reward, label='Auxiliary')
plt.scatter(eps, norm_ep_env_reward + norm_ep_aux_reward, label='Total')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig( f'{fig_dir}dqn_reward.png')
plt.figure()
plt.scatter(eps, ep_loss)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.savefig( f'{fig_dir}dqn_loss.png')
plt.show()
