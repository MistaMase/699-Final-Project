import numpy as np


class Goal:
    def __init__(self, name, color, goal_state, target_states, rewards):
        self.name = name
        self.color = color
        self.goal_state = np.array(goal_state)
        self.target_states = np.array(target_states)
        self.rewards = np.array(rewards)
