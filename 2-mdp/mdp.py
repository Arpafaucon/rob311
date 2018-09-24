#!/usr/bin/python3
#coding: utf-8

"""


arena
4       5       6
1       2       (3)

reward
-.1     -.1     -.1
-.1     -.1     1

move dynamics
        0.9
0.05            0.05

N E S W D

"""
ARENA = [[1, 2, 3], [4, 5, 6]]
ARENA_COORDINATES = {ARENA[lin][col]:(lin, col) for lin in range(2) for col in range(3)}
REWARD = [-0.1, -0.1, 1, -0.1, -0.1, -0.1]



ACTION_NAMES = {
    0: 'NORTH',
    1: 'EAST',
    2: 'SOUTH',
    3: 'WEST',
    4: 'STILL',
}

ALLOWED_ACTIONS = {
    1: [0, 1, 4],
    2: [0, 1, 3, 4],
    3: [4],
    4: [1, 2, 4],
    5: [1, 2, 3,  4],
    6: [2, 3, 4],
}

def uncertain_move(cell_from, action):
    if not action in ALLOWED_ACTIONS[cell_from]:
        return (False, [])
    if ACTION_NAMES[action] == 'STILL':
        return True, [(cell_from, 1)]
    else:
        lin, col = ARENA_COORDINATES[cell_from]
        