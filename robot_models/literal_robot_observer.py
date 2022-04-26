import numpy as np


class LiteralRobotObserver(object):
    def __init__(self, human_handler):
        self.human_handler = human_handler
        self.traj = []
        self.pi_traj_g = np.ones(len(self.human_handler.candidate_envs), dtype=float)
        self.pi_g_traj = np.ones(len(self.human_handler.candidate_envs), dtype=float) / len(self.human_handler.candidate_envs)

    def update(self, a_t, s_t):
        """
        Update the belief over goals based on the trajectory observed so far.
        :param a_t: human action D(u) derived from human input
        :param s_t: current state of environment

        Note:
        self.pi_g_traj: array of beliefs over goals.
        self.human_handler.policy.goal_literal_human_policies: list of the goal_policies (for each goal)
        For each goal_policy,
            goal_policy.get_q(s_t, a_t) returns the Q_{g}(x_{t}, u_{t})
            goal_policy.get_v(s_t) returns V_{g}(x_{t})


        Save the updated belief b_{t}(g) to self.pi_g_traj
        self.pi_traj_g can be used to store the product of probabilities of the observed human inputs
        """

        # append current human action and state
        self.traj.append([a_t, s_t])

        # ------------------------------------------------------------------------------------------------------------ #
        #                                     Update belief over goals (pi_g_traj)
        # ------------------------------------------ FILL YOUR CODE HERE --------------------------------------------- #

        # iterate over all goal policies
        for goal_num, goal_policy in enumerate(self.human_handler.policy.goal_literal_human_policies):

            # q_g for the current state and human action
            q_g = goal_policy.get_q(s_t, a_t)

            # v_g for the current state
            v_g = goal_policy.get_v(s_t)

            # probability of input given goal
            pi_u_given_x_and_g = np.exp(q_g) / np.exp(v_g)

            # probability of trajectory given goal
            self.pi_traj_g[goal_num] = self.pi_traj_g[goal_num] * pi_u_given_x_and_g

        # update self.pi_g_traj
        self.pi_g_traj = (self.pi_traj_g * self.pi_g_traj) / np.sum(self.pi_traj_g * self.pi_g_traj)

        # ------------------------------------------------------------------------------------------------------------ #

        self.pi_g_traj = np.clip(self.pi_g_traj, 0.05, 0.95)

    def get_result(self):
        return_str = []
        for i, goal_policy in enumerate(self.human_handler.policy.goal_literal_human_policies):
            return_str.append('Goal {} prob: {:.2f}'.format(i, self.pi_g_traj[i]))
        return return_str
