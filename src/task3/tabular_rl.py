import json
import random

import numpy as np
import gridworld as gw

import gridworld_vis as gv
import matplotlib.pyplot as plt
import seaborn as sb


def sarsa(world, ep_num, prob, step_size=1, max_step=80):

    # Initiate lookup table Q
    # Conventional: First dim: current_state_x, Second dim: current_state_y
    # Third dim: 0 - left, 1 - right, 2 - up, 3 - down
    Q = np.zeros((16, 16, 4))
    moves = []
    returns = []

    for i in range(ep_num):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False
        world.reset()
        S = world.current_state
        current_step = 0
        moves = []
        ret = 0

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
            # Retriving observed data and print info
            moves.append((new_state.x-S.x,new_state.y-S.y))
            ret += reward
            current_step += 1
            # Step 5
            action = new_action
            S = new_state
        
        if isinstance(S, gw.GoalCell) and not isinstance(S, gw.PitCell):
            returns.append((i,ret))
    return moves, returns


def q_learn(world, ep_num, prob, step_size=1, max_step=100):
    Q = np.zeros((16, 16, 4))
    returns = []
    moves = []

    for i in range(ep_num):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False
        world.reset()
        S = world.current_state
        current_step=0
        moves = []
        ret = 0

        while not done and current_step < max_step:
            # Step 1
            A = choose_action(world, Q, prob)
            # Step 2
            new_state, reward, done = world.step(A[0])
            # Step 3: Update Q
            lookup_values = Q[new_state.x, new_state.y, :]
            max_arg = max(lookup_values)
            Q[S.x, S.y, A[1]] += step_size * (reward + max_arg - Q[S.x, S.y, A[1]])
            # Retriving observed data and print info
            print(f'Current step: {current_step}, Current Cell: {S}, Next Dir: {A[0]}, Q: {Q[S.x, S.y, A[1]]}')
            print('.')
            ret += reward
            moves.append((new_state.x-S.x,new_state.y-S.y))
            current_step += 1
            # Step 4
            S = new_state
        
        if isinstance(S, gw.GoalCell) and not isinstance(S, gw.PitCell):
            returns.append((i,ret))
    return moves, returns


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


def value_iteration(world, discount=1.0, theta=1e-9, max_iter=1e4):
    # Initialize state-value function with zeros for each environment state
    print('---------------------------------')
    print('STARTING VALUE ITERATION')
    V = np.zeros(world.grid.shape)
    for i in range(int(max_iter)):
        # Early stopping condition
        delta = 0
        # Update each state
        for state in world.grid.flatten():
            if isinstance(state,gw.WallCell): continue
            # Do a one-step lookahead to calculate state-action values
            action_values = one_step_lookahead(world, state, V, discount)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_values)
            # Calculate change in value
            delta = max(delta, np.abs(V[state.y,state.x] - best_action_value))
            # Update the value function for current state
            V[state.y,state.x] = best_action_value
            # Check if we can stop
        if delta < theta:
                print(f'Value-iteration converged at iteration#{i}.')
                break
        print(f'Iter {i}, Delta {delta}')
    # Create a deterministic policy using the optimal value function
    policy = np.zeros((16, 16, 4))
    for state in world.grid.flatten():
            if isinstance(state,gw.WallCell): continue
            # One step lookahead to find the best action for this state
            action_values = one_step_lookahead(world, state, V, discount)
            # Select best action based on the highest state-action value
            best_action = np.argmax(action_values)
            # Update the policy to perform a better action at a current state
            policy[state.x, state.y, best_action] = 1.0
    return policy, V


def one_step_lookahead(world, state, V, discount_factor):
    nA = 4
    action_values = np.zeros(nA)
    for action_num in range(nA):
        action = action_num_to_text(action_num)
        proposed_state = state.step(action)
        if proposed_state.allow_enter(state, action):
            next_state = proposed_state
        else:
            next_state = state
        reward = world.reward_class.reward_f(state, action, next_state)
        p = world.p(next_state, reward, state, action)
        action_values[action_num] += p * (reward + discount_factor * V[next_state.y, next_state.x])
    return action_values


def walk_with_policy(world, policy, max_step=60):
    done = False
    current_step = 0
    moves = []
    world.reset()
    S = world.current_state
    while not done and current_step < max_step:
        A = choose_action(world, policy, 0)
        new_state, reward, done = world.step(A[0])
        moves.append((new_state.x-S.x,new_state.y-S.y))
        current_step += 1
        # Step 4
        S = new_state
    return moves



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


def plot_ret_eps(returns):
    plt.scatter(*zip(*returns))
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.title('Return on Episode')
    plt.show()


if __name__ == '__main__':
    world = gw.World.load_from_file('world.json')
    # actions, returns = sarsa(world,10000, 0.1, step_size=0.5, max_step=60)
    # actions, returns = q_learn(world,10000, 0.01, step_size=1, max_step=60)
    # visualize_gridworld(world,actions)
    # plot_ret_eps(returns)

    # world.reward_class = gw.ThirdReward
    policy,V = value_iteration(world)
    sb.heatmap(V,annot=True)
    plt.show()
    # visualize_gridworld(world,walk_with_policy(world, policy))