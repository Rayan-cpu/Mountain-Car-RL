import gymnasium as gym
import agents 
import numpy as np
import h5py as h5


fig_dir = '../figs/'
output_dir = '../output/'
output_file = 'dqn_agent'
env = gym.make('MountainCar-v0')
#env = gym.make('MountainCar-v0', render_mode='rgb_array')

agent = agents.RandomAgent()
#agent_dqn = agents.DQNAgentHeuristic( degree=3, frac=1.e-1 )
agent_dqn = agents.DQNAgentRND( reward_factor=1.0 )


n_eps = 100
sampling = n_eps // 20
count = np.zeros(n_eps)
norm_ep_env_reward = np.zeros(n_eps)
norm_ep_aux_reward = np.zeros(n_eps)
ep_loss = np.zeros(n_eps)

for i in range(n_eps):
    count[i], norm_ep_env_reward[i], norm_ep_aux_reward[i], ep_loss[i] = agent_dqn.run_episode(env)
    if i % sampling == 0:
        print(i/n_eps * 100 , '% done')

# save the data to a .h5 file
with h5.File(f'{output_dir}{output_file}.h5', 'w') as f:
    f.create_dataset('eps', data=range(n_eps))
    f.create_dataset('count', data=count)
    f.create_dataset('norm_ep_env_reward', data=norm_ep_env_reward)
    f.create_dataset('norm_ep_aux_reward', data=norm_ep_aux_reward)
    f.create_dataset('ep_loss', data=ep_loss)


'''
done = False
episode_reward = 0
states = np.zeros((1000, 2))
states[0] = state
count = 0
images = []
while not done:
    action = agent.select_action(state,env) 
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.observe(state, action, next_state, reward)
    agent.update()

    episode_reward += reward
    state = next_state
    states[count+1] = state

    done = terminated or truncated
    count += 1
    images.append(env.render())
env.close()



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plot the images with a .1 second delay

fig = plt.figure()
# updates the data and graph
def update(frame):
    # creating a new graph or updating the graph
    plt.imshow(frame)
 
anim = FuncAnimation(fig, update, frames = images, cache_frame_data=False, interval=50)
plt.show()
'''

'''
fig = plt.figure()
for i in range(len(images)):
    plt.imshow(images[i])
    plt.pause(.1)
    plt.draw()
'''
