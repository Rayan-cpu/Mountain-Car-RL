import random

# does an agent have an environment? 
class RandomAgent:
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        # Select a random action from the available actions
        return random.choice(state.available_actions)

    def update(self):
        pass


done = False
state, _ = env.reset()
episode_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, terimnated, truncated, _ = env.step(action)

    episode_reward += reward

    state = next_state
    done = terminated or truncated