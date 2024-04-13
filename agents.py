import random

# does an agent have an environment? 
class RandomAgent:
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state, env): # shouldn't this have access to the environment ? -> could then use action = env.action_space.sample()
        return env.action_space.sample()
    #    return random.choice(state.available_actions)

    def update(self):
        pass
