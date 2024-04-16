import random

# does an agent have an environment? 
class RandomAgent:
    def __init__(self):
        self.actions = [0,1,2] # we can move left, stay or move right
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state): # no access to the environment as there will be no illegal actions
        return random.choice( self.actions )

    def update(self):
        pass
