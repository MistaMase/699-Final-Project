from itertools import product
import simplejson
import random

import numpy as np

from grid_world.goal import Goal
from dps.dp import DP


class GridWorld(DP):
    """
    This class is to define the rules of the games.
    [0, 0] - - - - - - - - - - - - >x
    |
    |
    |
    |
    |
    |
    |
    |
    V y
    """

    def __init__(self, game_description_file, goal_idx=-1, target_idx=-1, deterministic=False):
        super(DP, self).__init__()
        self.is_deterministic = deterministic
        with open(game_description_file, 'r') as json_f:
            game_description = simplejson.load(json_f)
            self.num_grid = game_description['num_grid']
            self.robot_state = np.array(game_description['robot_state'])

            self.goals = []
            self.ditches = np.array(game_description['ditches']['states'])
            self.ditch_rewards = np.array(game_description['ditches']['rewards'])
            self.obstacles = np.array(game_description['obstacles'])
            self.initialize_goals(game_description['goals'])  # all possible goals

            self.actions = [0, 1, 2, 3]  # 0: up, 1: right, 2: down, 3: left
            self.delta_actions = np.array([np.array((0, -1)), np.array((1, 0)),
                                           np.array((0, 1)), np.array((-1, 0))])  # s' = s + delta_a
            self.construct_env(goal_idx, target_idx)

    def get_init_state(self):
        return self.init_int_state

    def construct_env(self, goal_idx=-1, target_idx=-1):
        """set mdp, r, t, """
        # define grid states and features
        self.width = self.num_grid
        self.height = self.num_grid
        self.feature_states = np.array(
            list(product(range(self.height), range(self.width))))  # the coordinate states (x, y)
        self.int_states = np.array(list(range(self.height * self.width)))  # use integers to represent states

        # robot start state
        self.init_feature_state = self.robot_state.copy()
        self.init_int_state = self.feature_to_int(self.init_feature_state)

        # sample goal index
        if goal_idx == -1 and target_idx == -1:
            self.goal_idx = random.randint(0, len(self.goals) - 1)
            self.target_idx = random.randint(0, self.goals[goal_idx].target_states.shape[0] - 1)
        elif goal_idx == -1 or target_idx == -1:
            raise ValueError
        else:
            self.goal_idx = goal_idx
            self.target_idx = target_idx

        # define rewards
        self.rewards = np.ones_like(self.int_states) * -1

        # define goal state reward
        self.target_feature_state = self.goals[self.goal_idx].target_states[self.target_idx]  # here goal is the target
        self.target_int_state = self.feature_to_int(self.target_feature_state)
        self.rewards[self.target_int_state] = self.goals[self.goal_idx].rewards[self.target_idx]

        # define cost of ditches
        for ditch_idx, ditch in enumerate(self.ditches):
            ditch_int_state = self.feature_to_int(ditch)
            self.rewards[ditch_int_state] = self.ditch_rewards[ditch_idx]

    def transition(self, int_s, int_a):
        drift_a = self.actions[int_a - 1]
        if self.is_deterministic:
            act = int_a
        else:
            act = np.random.choice([int_a, drift_a], 1, p=[0.9, 0.1])[0]

        if act not in self.available_actions(int_s):
            return int_s
        feature_s = self.int_to_feature(int_s)
        delta_action = self.delta_actions[act]
        feature_ns = feature_s + delta_action
        int_ns = self.feature_to_int(feature_ns)
        return int_ns

    def possible_transitions(self, int_s, int_a):
        """
        Returns all transitions that can occur from state int_s after taking action int_a.
        In this grid world, there is a 80% chance of going in the direction of int_a,
        and a 20% chance of going in the direction left of int_a.

        return: List of tuples. Each tuple contains the resulting state and the probability of reaching it.
        """

        feature_s = self.int_to_feature(int_s)

        # 90% chance of moving in the int_a direction
        if int_a not in self.available_actions(int_s):
            int_ns = int_s
        else:
            delta_action = self.delta_actions[int_a]
            feature_ns = feature_s + delta_action
            int_ns = self.feature_to_int(feature_ns)

        # 10% chance of drifting left
        drift_a = self.actions[int_a - 1]
        if drift_a not in self.available_actions(int_s):
            drift_ns = int_s
        else:
            delta_action = self.delta_actions[drift_a]
            feature_ns = feature_s + delta_action
            drift_ns = self.feature_to_int(feature_ns)

        if self.is_deterministic:
            return [(int_ns, 1.0)]
        else:
            return [(int_ns, 0.9), (drift_ns, 0.1)]

    def available_actions(self, int_s):
        """The agent can't go through the wall"""
        feature_s = self.int_to_feature(int_s)
        available_actions_set = list()
        if feature_s[0] > 0 and [feature_s[0] - 1, feature_s[1]] not in self.obstacles.tolist():
            available_actions_set.append(3)
        if feature_s[0] < self.num_grid - 1 and [feature_s[0] + 1, feature_s[1]] not in self.obstacles.tolist():
            available_actions_set.append(1)
        if feature_s[1] > 0 and [feature_s[0], feature_s[1] - 1] not in self.obstacles.tolist():
            available_actions_set.append(0)
        if feature_s[1] < self.num_grid - 1 and [feature_s[0], feature_s[1] + 1] not in self.obstacles.tolist():
            available_actions_set.append(2)

        return available_actions_set

    def is_terminal(self, int_s):
        """
        This returns if the state is terminal
        """
        return int_s == self.target_int_state

    def reward(self, int_s=None, int_a=None, int_ns=None):
        """The reward is given by the current state"""
        return self.rewards[int(int_s)]

    def feature_to_int(self, feature_state):
        int_state = feature_state[0] * self.height + feature_state[1]
        return int_state

    def int_to_feature(self, int_state):
        x = int(np.floor(int_state / self.height))
        y = int_state % self.height
        return np.array([x, y])

    def initialize_goals(self, goal_description):
        for goal in goal_description:
            self.goals.append(Goal(goal['name'],
                                   goal['color'],
                                   goal['goal_state'],
                                   goal['target_states'],
                                   goal['rewards']))
