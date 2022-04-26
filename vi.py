import numpy as np


def value_iteration(env, max_iterations=500, delta=0.001, gamma=0.9, if_softmaximum=True):
    """
    Value Iteration algorithm.
    :param env: grid world environment
    :param max_iterations: maximum iterations to run value iteration for
    :param delta: threshold for change in values to check for convergence
    :param gamma: discount factor

    env.available_actions(state): List of actions that can be executed from that state
    env.possible_transitions(state, available_action): List of resulting state tuples after executing available action.
                                                  Each tuple = (next state, probability of transitioning to that state)
    env.reward(state): Reward received in the state.
    """
    valid_states = [s for s in env.int_states if env.int_to_feature(s).tolist() not in env.obstacles.tolist()]
    vf = {s: 0 for s in valid_states}  # values
    qf = {s: {a: 0 for a in env.available_actions(s)} for s in valid_states}  # length of state * length of actions
    op_actions = {s: 0 for s in valid_states}  # optimal actions

    # Loop until maximum iterations
    for i in range(max_iterations):
        vf_temp = {s: 0 for s in valid_states}

        # Iterate over all states
        for j_state in vf:
            available_actions = env.available_actions(j_state)

            # Check if terminal state
            if j_state == env.target_int_state:  # keep the value function of the target 0
                vf_temp[j_state] = 1
                for k_action in available_actions:
                    qf[j_state][k_action] = 1
                op_actions[j_state] = k_action
                continue

            # Iterate over all actions to find the best action
            max_action = -1
            max_action_val = -np.inf
            for k_action in available_actions:
                #  q = r(s, a) + v(t(s, a))
                possible_ns = env.possible_transitions(j_state, k_action)
                qf[j_state][k_action] = env.reward(j_state)
                for int_ns, prob_ns in possible_ns:
                    qf[j_state][k_action] += prob_ns * (gamma * vf[int(int_ns)])

                # Select max value v = max_a q(s, a)
                if max_action_val < qf[j_state][k_action]:
                    max_action = k_action
                    max_action_val = qf[j_state][k_action]

            # Update the value of the state
            if if_softmaximum:
                q_vector = []
                for a in qf[j_state]:
                    q_vector.append(qf[j_state][a])
                vf_temp[j_state] = softmaximum(np.array(q_vector))
            else:
                vf_temp[j_state] = max_action_val

            # Simultaneously store the best action for the state
            op_actions[j_state] = max_action

        # After iterating over all states check if values have converged
        np_v = []
        np_v_temp = []
        for s in vf:
            np_v.append(vf[s])
            np_v_temp.append(vf_temp[s])
        np_v = np.array(np_v)
        np_v_temp = np.array(np_v_temp)
        change = np.linalg.norm((np_v - np_v_temp))
        vf = vf_temp
        if change < delta:
            print("VI converged after %d iterations" % (i))
            break

    if change >= delta:
        print("VI did not converge after %d iterations (delta=%.2f)" % (i, change))

    return vf, qf, op_actions


def value_iteration_qmdp(env, human_act, max_iterations=500, delta=0.001, gamma=1.0, if_softmaximum=False):
    """
    We simplify the value iteration for robot by not considering the difference to human input.
    :param env: grid world environment
    :param human_act: D(u)
    :param max_iterations: maximum iterations to run value iteration for
    :param delta: threshold for change in values to check for convergence
    :param gamma: discount factor
    """
    valid_states = [s for s in env.int_states if env.int_to_feature(s).tolist() not in env.obstacles.tolist()]
    vf = {s: 0 for s in valid_states}  # values
    qf = {s: {a: 0 for a in env.available_actions(s)} for s in valid_states}  # length of state * length of actions
    op_actions = {s: 0 for s in valid_states}  # optimal actions

    # Loop until maximum iterations
    for i in range(max_iterations):
        vf_temp = {s: 0 for s in valid_states}

        # Iterate over all states
        for j_state in vf:
            available_actions = env.available_actions(j_state)

            # Check if terminal state
            if j_state == env.target_int_state:  # keep the value function of the target 0
                vf_temp[j_state] = 1
                for k_action in available_actions:
                    qf[j_state][k_action] = 1
                op_actions[j_state] = k_action
                continue

            # Iterate over all actions to find the best action
            max_action = -1
            max_action_val = -np.inf
            for k_action in available_actions:
                #  q = r(s, a) + v(t(s, a))
                possible_ns = env.possible_transitions(j_state, k_action)
                qf[j_state][k_action] = env.reward(j_state)
                for int_ns, prob_ns in possible_ns:
                    qf[j_state][k_action] += prob_ns * (gamma * vf[int(int_ns)])

                # Select max value v = max_a q(s, a)
                if max_action_val < qf[j_state][k_action]:
                    max_action = k_action
                    max_action_val = qf[j_state][k_action]

            # Update the value of the state
            if if_softmaximum:
                q_vector = []
                for a in qf[j_state]:
                    q_vector.append(qf[j_state][a])
                vf_temp[j_state] = softmaximum(np.array(q_vector))
            else:
                vf_temp[j_state] = max_action_val

            # Simultaneously store the best action for the state
            op_actions[j_state] = max_action

        # After iterating over all states check if values have converged
        np_v = []
        np_v_temp = []
        for s in vf:
            np_v.append(vf[s])
            np_v_temp.append(vf_temp[s])
        np_v = np.array(np_v)
        np_v_temp = np.array(np_v_temp)
        change = np.linalg.norm((np_v - np_v_temp))
        vf = vf_temp
        if change < delta:
            print("VI converged after %d iterations" % (i))
            break

    if change >= delta:
        print("VI did not converge after %d iterations (delta=%.2f)" % (i, change))

    return vf, qf, op_actions


def softmaximum(f):
    """ log(\sum_x e^{f(x)}) """
    return np.log(np.sum(np.exp(f)))
