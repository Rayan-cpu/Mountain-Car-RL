import gymnasium as gym

env = gym.make('MountainCar-v0', render_mode='human')

observation, info = env.reset()

for _ in range(2500):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

'''
done = False
state, _ = env.reset()
episode_reward = 0
states = np.zeros((1000, 2))
states[0] = state
while not done:
    action = env.action_space.sample()
    next_state, reward, terimnated, truncated, _ = env.step(action)

    episode_reward += reward

    state = next_state
    states[env.current_step] = state
    done = terminated or truncated

'''

