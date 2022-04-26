import numpy as np
from robot_models.literal_robot_policy import LiteralRobotPolicy
from robot_models.literal_robot_observer import LiteralRobotObserver


class RobotHandler(object):
    def __init__(self, human_handler):
        self.candidate_envs = human_handler.candidate_envs
        self.observer = LiteralRobotObserver(human_handler)

    def get_cmd(self, a_t, s_t):
        policy = LiteralRobotPolicy(self.candidate_envs, a_t)
        available_actions = self.candidate_envs[0][0].available_actions(s_t)

        # Iterate over all actions to find the best action
        max_action = -1
        max_action_val = -np.inf
        for k_action in available_actions:
            q = 0
            for i in range(len(policy.goal_literal_robot_policies)):
                q += self.observer.pi_g_traj[i] * \
                     policy.goal_literal_robot_policies[i].target_literal_robot_policies[0].qf[s_t][k_action]

            # Select max value v = max_a q(s, a)
            if max_action_val < q:
                max_action = k_action
                max_action_val = q

        return max_action

    def update(self, a_t, s_t):
        """Used to update the belief over the human goal"""
        self.observer.update(a_t, s_t)

    def get_result(self):
        return self.observer.get_result()
