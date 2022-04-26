"""DP stands for decision process."""


class DP(object):
    def __init__(self):
        pass

    def is_terminal(self, s):
        raise NotImplementedError

    def transition_dist(self, s, a):
        raise NotImplementedError

    def transition(self, s, a, d=False):
        raise NotImplementedError

    def reward(self, s=None, a=None, ns=None):
        raise NotImplementedError

    def available_actions(self, s):
        raise NotImplementedError

    def get_init_state(self):
        raise NotImplementedError
