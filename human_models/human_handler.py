from human_models.literal_human_policy import LiteralHumanPolicy


class HumanHandler(object):
    def __init__(self, real_goal_idx, candidate_envs):
        self.candidate_envs = candidate_envs
        self.real_goal_idx = real_goal_idx  # sample a goal
        self.policy = LiteralHumanPolicy(self.candidate_envs)

    def get_cmd(self, int_s):
        human_cmd = self.policy.goal_literal_human_policies[self.real_goal_idx].get_cmd(int_s)
        return human_cmd

    def run(self, real_goal_idx, real_target_idx):
        #  generate one possible trajectory for the given goal idx and target idx
        traj = []
        init_int_state = self.candidate_envs[real_goal_idx][real_target_idx].init_int_state
        current_int_s = init_int_state
        while current_int_s != self.candidate_envs[real_goal_idx][real_target_idx].target_int_state:
            #  select action
            a = self.policy.goal_literal_human_policies[real_goal_idx].target_literal_human_policies[real_target_idx].get_cmd(current_int_s)
            ns = self.candidate_envs[real_target_idx][real_target_idx].transition(current_int_s, a)
            traj.append((current_int_s, a))
            current_int_s = ns
        traj.append((current_int_s, '%'))  # the end operator
        return traj

