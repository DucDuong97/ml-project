import json
import random

import numpy as np

import gridworld_vis as gv
import gridworld as gw


def sarsa(world, ep_num, prob, step_size=1, max_step=80):

    # Initiate lookup table Q
    # Conventional: First dim: current_state_x, Second dim: current_state_y
    # Third dim: 0 - left, 1 - right, 2 - up, 3 - down
    Q = np.zeros((16, 16, 4))

    for i in range(ep_num):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False
        world.reset()
        S = world.current_state
        current_step = 0
        moves = []

        # Step 1
        action = choose_action(world, Q, prob)
        while not done and current_step < max_step:
            # Step 2
            new_state, reward, done = world.step(action[0])
            # Step 3
            print(f'Current step: {current_step}, Current Cell: {S}, Next Dir: {action[0]}, Q: {Q[S.x, S.y, action[1]]}')
            print('.')
            new_action = choose_action(world, Q, prob)
            # Step 4: Update Q
            Q[S.x, S.y, action[1]] += step_size * (reward + Q[new_state.x, new_state.y, new_action[1]] - Q[S.x, S.y, action[1]])
            # Step 5
            moves.append((new_state.x-S.x,new_state.y-S.y))
            action = new_action
            S = new_state
            current_step += 1
    return moves


def q_learn(world, ep_num, prob, step_size=1, max_step=100):
    Q = np.zeros((16, 16, 4))

    for i in range(ep_num):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False
        world.reset()
        S = world.current_state
        current_step=0
        moves = []

        while not done and current_step < max_step:
            # Step 1
            A = choose_action(world, Q, prob)
            # Step 2
            new_state, reward, done = world.step(A[0])
            # Step 3: Update Q
            lookup_values = Q[new_state.x, new_state.y, :]
            max_arg = max(lookup_values)
            Q[S.x, S.y, A[1]] += step_size * (reward + max_arg - Q[S.x, S.y, A[1]])

            # Step 4
            print(f'Current step: {current_step}, Current Cell: {S}, Next Dir: {A[0]}, Q: {Q[S.x, S.y, A[1]]}')
            print('.')
            moves.append((new_state.x-S.x,new_state.y-S.y))
            S = new_state
            current_step += 1
    return moves


def choose_action(world, Q, prob):
    choice = random.choices(["random_action", "argmax_action"], [prob, 1 - prob], k=1)[0]
    if choice == "random_action":
        A = random.choice(['left', 'right', 'up', 'down'])
        action_num = action_text_to_num(A)
        print(f'Random Choice, {A}')
    else:
        lookup_values = Q[world.current_state.x, world.current_state.y, :]
        max_lookup_value = max(lookup_values)
        action_num = list(lookup_values).index(max_lookup_value)
        A = action_num_to_text(action_num)
        print(f'Maximum Choice, {lookup_values}, {A}')
    return A, action_num


def action_text_to_num(action):
    if action == 'left':
        num = 0
    elif action == 'right':
        num = 1
    elif action == 'up':
        num = 2
    else:
        num = 3
    return num


def action_num_to_text(action):
    if action == 0:
        text = 'left'
    elif action == 1:
        text = 'right'
    elif action == 2:
        text = 'up'
    else:
        text = 'down'
    return text

def visualize_gridworld(world,actions):
    def tile2classes(x, y):
        cell = world.get_state(x,15 - y)
        if isinstance(cell, gw.PitCell):
            return "lava"
        if isinstance(cell, gw.StartCell):
            return "recharge"
        if isinstance(cell, gw.SwampCell):
            return "water"
        if isinstance(cell, gw.WallCell):
            return "line"
        if isinstance(cell, gw.ArrowCell):
            return "dry"
        if isinstance(cell, gw.GoalCell):
            return "recharge"
        return "normal"  
    svg = gv.gridworld(n=16, tile2classes=tile2classes, actions=actions)
    svg.saveas("../../report/task3/figures/grid.svg", pretty=True)


if __name__ == '__main__':
    world = gw.World.load_from_file('world.json')
    # actions = sarsa(world,10000, 0.1, step_size=0.5, max_step=60)
    actions = q_learn(world,1, 0.1, step_size=1, max_step=60)
    visualize_gridworld(world,actions)