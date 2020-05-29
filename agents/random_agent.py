import random


class RandomAgent:
    def __init__(self):
        pass

    def act(self, state_obs, actions):
        return random.choice(range(len(actions)))

    def add_observation(self, frame, action, next_frame, reward):
        pass
