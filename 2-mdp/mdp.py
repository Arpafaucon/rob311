#!/usr/bin/python3
# coding: utf-8

"""
author: Grégoire ROUSSEL

TP2 : Markov Decision Process


# Definitions

arena
4       5       6
1       2       (3)

reward
-.1     -.1     -.1
-.1     -.1     1

move dynamics
        0.9
0.05            0.05
"""
# ARENA = [[1, 2, 3], [4, 5, 6]]
ARENA_SIZE = 6
REWARD = [-0.1, -0.1, 1, -0.1, -0.1, -0.1]


ACTION_NAMES = {
    0: 'NORTH',
    1: 'EAST',
    2: 'SOUTH',
    3: 'WEST',
    4: 'STILL',
}

# how to find the next cell name when applying action
# new_cell = cell + ACTION_DELTA[action]
ACTION_DELTA = {
    0: 3,
    1: 1,
    2: -3,
    3: -1,
    4: 0
}

# actions allowed for each cell
ALLOWED_ACTIONS = {
    1: [0, 1, 4],
    2: [0, 1, 3, 4],
    3: [4],
    4: [1, 2, 4],
    5: [1, 2, 3,  4],
    6: [2, 3, 4],
}


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

# print color.BOLD + 'Hello World !' + color.END



def simple_move(cell_from, action):
    """
    Compute destination cell after action
    
    Args:
        cell_from (int): cell name
        action (int): action id
    
    Returns:
        int: new cell name
    """
    return cell_from + ACTION_DELTA[action]


def uncertain_move(cell_from, action):
    """
    For a starting cell and a chosen action, list all possible arrival cells with the associated probability
    
    Args:
        cell_from (int): name of cell, 1-6
        action (int): index of action, 0-4
    
    Returns:
        (bool, list): tuple (action_is_allowed, [(destination_cell_1, probability), ...])
    """
    if not action in ALLOWED_ACTIONS[cell_from]:
        return (False, [])
    if ACTION_NAMES[action] == 'STILL':
        return True, [(cell_from, 1)]
    else:
        # lin, col = ARENA_COORDINATES[cell_from]
        action_list = []
        # main action
        dest_cell = simple_move(cell_from, action)
        action_list.append((dest_cell, 0.9))
        # side action left
        stay_still_proba = 0
        side_action_left = (action - 1) % 4
        if side_action_left in ALLOWED_ACTIONS[cell_from]:
            # register action
            action_list.append(
                (simple_move(cell_from, side_action_left), 0.05))
        else:
            stay_still_proba += 0.05
        # side action right
        side_action_right = (action + 1) % 4
        if side_action_right in ALLOWED_ACTIONS[cell_from]:
            # register action
            action_list.append(
                (simple_move(cell_from, side_action_right), 0.05))
        else:
            stay_still_proba += 0.05
        # if some action were impossible , there is a non-null proba of staying at the same place
        if stay_still_proba:
            action_list.append((cell_from, stay_still_proba))

        return True, action_list


def action_value(cell, action, prev_board_values, gamma):
    """
    Return the expected value of the cell when using a given action
    
    Args:
        cell (int): cell name
        action (int): action index
        prev_board_values ([int]): values of all cells at the previous iteration
        gamma (float): discount factor
    
    Raises:
        NotImplementedError: if action is not permitted for this cell
    
    Returns:
        float: expected reward for the given (cell, action)
    """
    possible, destination_list = uncertain_move(cell, action)
    if not possible:
        raise NotImplementedError('illegal action')
    # cell_ix = cell - 1

    avg_reward = sum((proba_to*(REWARD[cell_to - 1] + gamma * prev_board_values[cell_to - 1])
                      for cell_to, proba_to in destination_list))
    return avg_reward


def cell_value(cell, prev_board_values, gamma):
    """
    Return the best action for a given cell, and the expected reward
    
    Args:
        cell (int): cell name
        prev_board_values ([int]): values of all cells at the previous iteration
        gamma (float): discount factor
    
    Returns:
        (float, int): (reward, action) that yields the best reward
    """
    action_outcomes = []
    for action in ALLOWED_ACTIONS[cell]:
        action_outcomes.append(
            (action_value(cell, action, prev_board_values, gamma), action))
    best_action = max(action_outcomes, key=lambda k: k[0])
    return best_action


def next_board_values(prev_board_values, gamma):
    """
    Build the value of all cells for the next iteration
    
    Args:
        prev_board_values ([int]): current cell values
        gamma (float): discount factor
    
    Returns:
        [float]: new values of the cells
    """
    next_board = []
    policy = []
    for cell in range(1, 7):
        cell_policy = cell_value(cell, prev_board_values, gamma)
        next_board.append(cell_policy[0])
        policy.append(ACTION_NAMES[cell_policy[1]])
    return next_board, policy


def rms_board(b1, b2):
    """
    Distance between two set of cell values
    
    Args:
        b1 ([float]): 
        b2 ([float]): 
    
    Returns:
        float: Root-Mean-Square distance between two sets
    """
    sq_sum = sum((b1[i] - b2[i])**2 for i in range(ARENA_SIZE))
    rms = (sq_sum ** 0.5)/ARENA_SIZE
    return rms


def print_board_values(board):
    for lin in [1, 0]:
        line = board[lin*3:]
        print('{:.3f}\t{:.3f}\t{:.3f}'.format(line[0], line[1], line[2]))

def print_policy(board_policy):
    for lin in [1, 0]:
        line = board_policy[lin*3:]
        print('{}\t{}\t{}'.format(line[0], line[1], line[2]))


def dbg_list_actions():
    for cell_from in range(1, 7):
        for action in range(5):
            print('{}.{}:  {}'.format(
                cell_from, ACTION_NAMES[action], uncertain_move(cell_from, action)))


# default values
GAMMA = 0.999
EPSILON = 1e-3
def value_iteration(gamma = GAMMA, epsilon=EPSILON):
    """
    Perform the value iteration solution

    Args:
        gamma (float, optional): Defaults to GAMMA. Discount
        epsilon (float, optional): Defaults to EPSILON. Distance min between two iteration to keep going on
    """
    board = [0]*ARENA_SIZE
    policy = []
    delta = 0
    done = False
    iteration_num = 0
    print('{}>>> Value Iteration for \u03b3 = {} ; \u03b5 = {}{}'.format(color.BOLD+color.DARKCYAN, gamma, epsilon, color.END))
    
    while not done:
        p_board, p_policy, p_delta = board, policy, delta
        iteration_num += 1
        nboard, policy = next_board_values(board, gamma)
        delta = rms_board(board, nboard)
        done = (delta <= epsilon)
        board = nboard
    print('{}convergence in {} steps (\u03b4={:.8f}) {} '.format(color.RED , iteration_num, delta, color.END))
    print('{}Value Map:{}'.format(color.UNDERLINE, color.END))
    print_board_values(board)
    print('{}Policy:{}'.format(color.UNDERLINE, color.END))
    print_policy(policy)
    print('\n')
    # print('{}{}\n'.format(color.PURPLE, color.END))
    print('{}previous state (step {},  \u03b4={:.6f}) {} '.format(color.PURPLE , iteration_num-1, p_delta, color.END))
    print('{}Value Map:{}'.format(color.UNDERLINE, color.END))
    print_board_values(p_board)
    print('{}Policy:{}'.format(color.UNDERLINE, color.END))
    print_policy(p_policy)
    print('\n')


def main():
    value_iteration(0.999, 1e-3)
    value_iteration(0.1, 1e-3)

if __name__ == '__main__':
    main()
