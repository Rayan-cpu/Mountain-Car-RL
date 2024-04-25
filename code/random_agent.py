import gymnasium as gym
import agents 
import numpy as np

fig_dir = '../figs/'

env = gym.make('MountainCar-v0')
#env = gym.make('MountainCar-v0', render_mode='rgb_array')

agent = agents.RandomAgent()
agent_dqn = agents.DQNAgent( heuristic=True )


n_eps = 500
count = np.zeros(n_eps)
episode_reward = np.zeros(n_eps)
episode_loss = np.zeros(n_eps)

for i in range(n_eps):
    count[i], episode_reward[i], episode_loss[i] = agent_dqn.run_episode(env)
    if i % 50 == 0:
        print(i/n_eps * 100 , '% done')


import matplotlib.pyplot as plt
plt.scatter(range(n_eps), count)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.savefig( f'{fig_dir}dqn_duration.png')
plt.figure()
plt.scatter(range(n_eps), episode_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig( f'{fig_dir}dqn_reward.png')
plt.figure()
print(episode_reward)
plt.scatter(range(n_eps), episode_loss)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.savefig( f'{fig_dir}dqn_loss.png')
plt.show()


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
