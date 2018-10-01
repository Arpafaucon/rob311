#!/usr/bin/python3
#coding: utf-8
"""
Partially Observable Markov Decision Process

author: Grégoire ROUSSEL

states :  names 1->8
        indices 0->7
actions 'LRUD'
observation 'ULTX'

    3   5
    |   |
1 - 2 - 4 - 7
        |   |
        6   8
"""
import numpy as np

TRANS_AR_D = np.ones((8, 8))*.05

#line : origin
#col: destination
TRANS_R = np.array([[0.15,  0.8,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.15,  0.05,  0.8,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.05,  0.15,  0.05,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.15,  0.05,  0.05,  0.8,  0.05],
                    [0.05,  0.05,  0.05,  0.05,  0.15,  0.05,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.05,  0.05,  0.15,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.15,  0.8],
                    [0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.15]])

TRANS_D = np.array([[0.15,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.15,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.8,  0.15,  0.05,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.015,  0.05,  0.05,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.05,  0.15,  0.8,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.8,  0.05,  0.15,  0.05,  0.05],
                    [0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.15,  0.8],
                    [0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.15]])

TRANS_L = np.array([[0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.8, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.8, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.8, 0.05, 0.05, 0.15, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15]])

TRANS_U = np.array([[0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.15, 0.8, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.15, 0.8, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.8, 0.05, 0.15, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15]])

TRANS = {
    'U': TRANS_U,
    'D': TRANS_D,
    'L': TRANS_L,
    'R': TRANS_R
}

OBSERVATION = np.array([[0.9, 0.08, 0.01, 0.01],
                        [0.08, 0.85, 0.06, 0.01],
                        [0.01, 0.06, 0.88, 0.05],
                        [0.01, 0.01, 0.05, 0.93]])

ENVIRONMENT = 'UTUXUULU'
ENV_IX = {
    'U': 3,
    'L': 2,
    'T': 1,
    'X': 0
}

SE_0 = [0.05, .1, .1, .7, .005, .01, .005, .03]
SE_1 = np.ones(8) / 8

# log in the form (action, observation)
LOG = [('R', 'L'),
       ('D', 'U')]


class color:
    """
    Display purposes only
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# print(repr(OBSERVATION))


def normalize(state_vector):
    total = sum(state_vector)
    return state_vector / total


def see_obs_in_state(env_observed, state_vector):
    # total_proba = 0
    observed_proba = np.zeros(len(state_vector))
    for i_state in range(len(state_vector)):
        real_env = ENVIRONMENT[i_state]
        proba_obs_state = OBSERVATION[ENV_IX[env_observed], ENV_IX[real_env]]
        # total_proba += proba_obs_state * proba_state
        observed_proba[i_state] = proba_obs_state
    return observed_proba


def test_see_obs_in_space():
    print(see_obs_in_state('U', [.5, 0, 0, .5, 0, 0, 0, 0]))
    print(see_obs_in_state('X', [.5, 0, 0, .5, 0, 0, 0, 0]))


def update_state_vector(state_vector, action_taken):
    unnorm_new_state = np.dot(state_vector, TRANS[action_taken])
    # print(unnorm_new_state)
    new_state = normalize(unnorm_new_state)
    return new_state

def print_state(state_vector, precision=3, pretty_max=True):
    rounded = [round(p, precision) for p in state_vector]
    
    if pretty_max:
        perc75 = np.percentile(state_vector, 75)
    else:
        perc75 = 1
    pretty = [color.DARKCYAN+ repr(p) + color.END if p > perc75 else repr(p) for p in rounded]
    st = ' '.join(pretty)
    print(st)
# def iterate_


def iterate(state_vector, action_taken, enviromnent_observed):
    new_state = update_state_vector(state_vector, action_taken)
    state_after_observation = new_state * \
        see_obs_in_state(enviromnent_observed, new_state)
    norm_state = normalize(state_after_observation)
    return norm_state


def journey(initial_state_vector, log, case='0'):
    state_vector = initial_state_vector
    np.set_printoptions(precision=3)
    print('{}case: {}{}'.format(color.BOLD+color.PURPLE, case, color.END))
    print('Starting POMDP with')
    print_state(initial_state_vector)
    print('\n')
    for i, log_entry in enumerate(log):
        action, observation = log_entry
        print('>>> {}Step {} (action={}, observed={}){}'.format(
            color.RED,  i, action, observation, color.END))
        state_vector = iterate(state_vector, action, observation)
        print_state(state_vector)
    print('\n')


def main():
    journey(SE_0, LOG, 'Localized')
    journey(SE_18, LOG, 'Uniform Dist')
    # test_see_obs_in_space()


if __name__ == '__main__':
    main()
