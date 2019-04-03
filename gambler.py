from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


HyperParameter = namedtuple('HyperParameter', ('num_states', 'discount', 'theta', 'p_h'))


def compute_q_value(state, values, parameter: HyperParameter):
    """return q(state, action) for all actions"""
    actions = np.arange(min(state, parameter.num_states-state)+1)
    # successor state per (state, action, head)
    next_state_win = state + actions
    # successor state per (state, action, tail)
    next_state_loss = state - actions
    # q value per per (current state, action)
    q_values = parameter.p_h * (np.where(next_state_win == parameter.num_states, 1.0, 0.0) +
                                parameter.discount * values[next_state_win])
    q_values += (1-parameter.p_h) * (parameter.discount * values[next_state_loss])
    return q_values


def value_iteration(parameter: HyperParameter, out_file=''):
    states = list(range(1, parameter.num_states))
    values = np.zeros(parameter.num_states+1)
    delta = float('inf')
    value_per_sweep = []
    while delta > parameter.theta:
        value_copy = np.copy(values)
        for state in states:
            values[state] = max(compute_q_value(state, values, parameter))
        delta = np.max(np.abs(values - value_copy))
        value_per_sweep.append(value_copy)
        print(f'[{len(value_per_sweep)}]: delta={delta}')
    if out_file:
        plt.figure()
        for vs in value_per_sweep[1:]:
            plt.plot(states, vs[1:-1])
        plt.savefig(out_file)
    return values


def compute_solution(policy_out='policy.png', value_out=''):
    hp = HyperParameter(num_states=100, discount=1.0, theta=1e-2, p_h=0.4)
    values = value_iteration(hp)
    states = list(range(1, hp.num_states))
    policies = [np.argmax(compute_q_value(state, values, hp)) for state in states]
    plt.figure()
    plt.plot(states, policies)
    plt.savefig(policy_out)


def compute_value(state=51):
    hp = HyperParameter(num_states=100, discount=1.0, theta=1e-2, p_h=0.4)
    values = value_iteration(hp)
    q_value = compute_q_value(state, values, hp)
    print(q_value)


compute_solution()
