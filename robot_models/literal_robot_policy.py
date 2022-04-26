import numpy as np
from vi import value_iteration_qmdp, softmaximum


class TargetLiteralRobotPolicy:
    def __init__(self, target_state, human_action, candidate_env):
        self.target_state = target_state

        # run value iteration
        self.vf, self.qf, self.op_actions = value_iteration_qmdp(candidate_env, human_action)

        self.candidate_env = candidate_env

    def get_cmd(self, int_s, if_softmaximum=True):
        if not if_softmaximum:
            return self.op_actions[int_s]
        else:
            available_actions = self.candidate_env.available_actions(int_s)
            pi_a_g_s = {a: np.exp(self.qf[int_s][a]) for a in available_actions}
            partition = 0
            for a in pi_a_g_s:
                partition += pi_a_g_s[a]
            for a in pi_a_g_s:
                pi_a_g_s[a] = pi_a_g_s[a] / partition
            temp = np.random.uniform()
            current_state = 0
            for a in pi_a_g_s:
                if current_state <= temp <= current_state + pi_a_g_s[a]:
                    return a
                current_state += pi_a_g_s[a]
            raise ValueError


class GoalLiteralRobotPolicy:
    def __init__(self, candidate_env_lst, human_action):
        self.candidate_env_lst = candidate_env_lst
        self.goal = self.candidate_env_lst[0].goals[self.candidate_env_lst[0].goal_idx]
        self.target_literal_robot_policies = []
        for i, target_state in enumerate(self.goal.target_states):
            self.target_literal_robot_policies.append(
                TargetLiteralRobotPolicy(target_state, human_action, self.candidate_env_lst[i]))

    def get_cmd(self, int_s):
        best_policy = None
        best_value = -np.inf
        for target_policy in self.target_literal_robot_policies:
            if target_policy.vf[int_s] > best_value:
                best_value = target_policy.vf[int_s]
                best_policy = target_policy
        return best_policy.get_cmd(int_s)

    def get_q(self, int_s, int_a):
        """q_g = softmax_k q_k"""
        q_vector = []
        for target_policy in self.target_literal_robot_policies:
            q_vector.append(target_policy.qf[int_s][int_a])
        return softmaximum(np.array(q_vector))

    def get_v(self, int_s):
        """v_g = softmax_k v_k"""
        v_vector = []
        for target_policy in self.target_literal_robot_policies:
            v_vector.append(target_policy.vf[int_s])
        return softmaximum(np.array(v_vector))


class LiteralRobotPolicy:
    def __init__(self, candidate_envs, human_action):
        self.candidate_envs = candidate_envs  # [[env]]
        self.goal_literal_robot_policies = []
        for candidate_env_lst in self.candidate_envs:
            self.goal_literal_robot_policies.append(GoalLiteralRobotPolicy(candidate_env_lst, human_action))
