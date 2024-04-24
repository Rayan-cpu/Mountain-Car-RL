import gymnasium as gym
import agents 
import numpy as np

env = gym.make('MountainCar-v0', render_mode='rgb_array')
state, info = env.reset()
agent = agents.RandomAgent()
agent_dqn = agents.DQNAgent()

def episode_time(env, agent):
    state, info = env.reset()
    done = False
    episode_reward = 0
    episode_loss = 0.
    count = 0
    
    while not done:
        action = agent.select_action(state) 
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.observe(state, action, next_state, reward)
        if done :
            episode_loss = agent.update( done=True )
        else : 
            agent.update( done=done )

        episode_reward += reward
        #episode_loss += loss
        state = next_state
        count += 1
    return count, episode_reward, episode_loss

# run 100 episodes and plot the duration of each in a scatter plot
n_eps = 100
count = np.zeros(n_eps)
episode_reward = np.zeros(n_eps)
episode_loss = np.zeros(n_eps)
for i in range(n_eps):
    count[i], episode_reward[i], episode_loss[i] = episode_time(env, agent_dqn)


import matplotlib.pyplot as plt
plt.scatter(range(n_eps), count)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.figure()
plt.scatter(range(n_eps), episode_reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.figure()
print(episode_loss)
plt.scatter(range(n_eps), episode_loss)
plt.xlabel('Episode')
plt.ylabel('Loss')
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


'''

for _ in range(2500):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

'''

