import random
import os

import numpy as np
import gridworld as gw

import gridworld_vis as gv
import matplotlib.pyplot as plt
import seaborn as sb
from make_figures import PATH, FIG_WITDH, FIG_HEIGHT, FIG_HEIGHT_FLAT, setup_matplotlib


def sarsa(world, ep_num=1000, prob=1, decay_rate=0.99, step_size=1, max_step=100):

    # Initiate lookup table Q
    # Conventional: First dim: current_state_x, Second dim: current_state_y
    # Third dim: 0 - left, 1 - right, 2 - up, 3 - down
    Q = np.zeros((16, 16, 4))
    best_move = []
    returns = []

    for i in range(ep_num):
        prob *= decay_rate
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        world.reset()
        moves = []
        ret = 0

        # Step 1
        A = choose_action(world, Q, prob)
        for step in range(max_step):
            S = world.current_state
            # Step 2
            new_S, reward, done = world.step(A[0])
            # Step 3
            new_A = choose_action(world, Q, prob)
            # Step 4: Update Q
            Q[S.x, S.y, A[1]] += step_size * (reward + Q[new_S.x, new_S.y, new_A[1]] - Q[S.x, S.y, A[1]])
            # Step 5
            A = new_A
            # Retriving observed data and print info
            print(f'Current step: {step}, Current Cell: {S}, Next Dir: {A[0]}, Q: {Q[S.x, S.y, A[1]]}')
            print('.')
            moves.append((new_S.x-S.x,new_S.y-S.y))
            ret += reward
            if done: break
        S = world.current_state
        if isinstance(S, gw.GoalCell) and not isinstance(S, gw.PitCell):
            best_move = moves
            returns.append(ret)
    return best_move, returns


def q_learn(world, ep_num=2000, prob=1, decay_rate=0.99, step_size=1, max_step=100):
    Q = np.zeros((16, 16, 4))
    best_move = []
    moves = []
    returns = []

    for i in range(ep_num):
        prob *= decay_rate
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        world.reset()
        moves = []
        ret = 0

        new_state = None
        for step in range(max_step):
            # Step 1
            S = world.current_state
            A = choose_action(world, Q, prob)
            # Step 2
            new_state, reward, done = world.step(A[0])
            # Step 3: Update Q
            lookup_values = Q[new_state.x, new_state.y, :]
            Q[S.x, S.y, A[1]] += step_size * (reward + max(lookup_values) - Q[S.x, S.y, A[1]])
            # Retriving observed data and print info
            print(f'Current step: {step}, Current Cell: {S}, Next Dir: {A[0]}, Q: {Q[S.x, S.y, A[1]]}')
            print('.')
            ret += reward
            moves.append((new_state.x-S.x,new_state.y-S.y))
            if done: break
        S = world.current_state
        if isinstance(S, gw.GoalCell) and not isinstance(S, gw.PitCell):
            best_move = moves
            returns.append(ret)
    return best_move, returns


def choose_action(world, Q, prob):
    choice = random.choices(["random_action", "argmax_action"], [prob, 1 - prob], k=1)[0]
    if choice == "random_action":
        A = random.choice(['left', 'right', 'up', 'down'])
        action_num = action_text_to_num(A)
        # print(f'Random Choice, {A}')
    else:
        lookup_values = Q[world.current_state.x, world.current_state.y, :]
        action_num =  np.argmax(lookup_values)
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
        new_V = np.zeros(world.grid.shape)
        # Early stopping condition
        delta = 0
        # Update each state
        for state in world.grid.flatten():
            if isinstance(state,gw.WallCell):
                continue
            # Do a one-step lookahead to calculate state-action values
            action_values = one_step_lookahead(world, state, V, discount)
            # Select best action to perform based on the highest state-action value
            best_action_value = np.max(action_values)
            # Calculate change in value
            delta = max(delta, np.abs(V[state.y,state.x] - best_action_value))
            # Update the value function for current state
            new_V[state.y,state.x] = best_action_value
            # Check if we can stop
        V = new_V
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
        next_state = None
        for proposed_state in state.get_afterstates(action):
            if proposed_state.allow_enter(state, action):
                if proposed_state is next_state:
                    continue
                next_state = proposed_state
            else:
                if state is next_state:
                    continue
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


def visualize_gridworld(world,actions,name="grid"):
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
    svg.saveas(os.path.join(PATH, f"{name}.svg"), pretty=True)


def plot_ret_eps(returns,name="plot"):
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.scatter(range(len(returns)), returns)
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.title(f'Return on Episode: {name}')
    fig.tight_layout()
    plt.savefig(os.path.join(PATH, f'{name}.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    setup_matplotlib()

    world = gw.World.load_from_file('world.json')


    # b
    actions, returns = sarsa(world,ep_num=2000)
    visualize_gridworld(world,actions,"sarsa_path")
    plot_ret_eps(returns,"sarsa_eps_return")


    # c
    k1 = []
    k2 = []
    success_num = []
    avg_rets = []
    for i in np.linspace(0,1,10):
        _, returns = sarsa(world,ep_num=1000,prob=i,decay_rate=1)
        k1.append(i)
        success_num.append(len(returns))
        if len(returns) > 0:
            k2.append(i)
            avg_rets.append(sum(returns) / max(1,len(returns)))
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.plot(k1, success_num)
    plt.xlabel('Probability')
    plt.title('Success eps over 1000 runs (Sarsa)')
    plt.savefig(os.path.join(PATH, 'sarsa_succ_eps.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.plot(k2, avg_rets)
    plt.xlabel('Probability')
    plt.title('Average success reward (Sarsa)')
    plt.savefig(os.path.join(PATH, 'sarsa_avg_r.pdf'))
    plt.close(fig)
    

    # d
    actions, returns = q_learn(world,ep_num=1000)
    visualize_gridworld(world,actions,"q_learning_path")
    plot_ret_eps(returns,"q_learning_eps_return")

    k1 = []
    k2 = []
    success_num = []
    avg_rets = []
    for i in np.linspace(0,1,10):
        _, returns = q_learn(world,ep_num=1000,prob=i,decay_rate=1)
        k1.append(i)
        success_num.append(len(returns))
        if len(returns) > 0:
            k2.append(i)
            avg_rets.append(sum(returns) / max(1,len(returns)))
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.plot(k1, success_num)
    plt.xlabel('Probability')
    plt.title('Success eps over 1000 runs (Q Learning)')
    plt.savefig(os.path.join(PATH, 'q_learning_succ_eps.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    plt.plot(k2, avg_rets)
    plt.xlabel('Probability')
    plt.title('Average success reward (Q Learning)')
    plt.savefig(os.path.join(PATH, 'q_learning_avg_r.pdf'))
    plt.close(fig)


    # e
    world.reward_class = gw.SecondReward
    actions, returns = q_learn(world,ep_num=1000)
    visualize_gridworld(world,actions,"reward_2_q_learning_path")
    plot_ret_eps(returns,"r_2_q_learning_eps_return")

    world.reward_class = gw.ThirdReward
    actions, returns = q_learn(world,ep_num=1000)
    visualize_gridworld(world,actions,"reward_3_q_learning_path")
    plot_ret_eps(returns,"reward_3_q_learning_eps_return")


    # f
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    policy,V = value_iteration(world)
    sb.heatmap(V,annot=True)
    plt.title('Optimal value function (first reward)')
    plt.savefig(os.path.join(PATH, 'value_iter_reward_1.pdf'))
    plt.close(fig)
    visualize_gridworld(world,walk_with_policy(world, policy),"value_iter_reward_1")

    world.reward_class = gw.SecondReward
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    policy,V = value_iteration(world)
    sb.heatmap(V,annot=True)
    plt.title('Optimal value function (second reward)')
    plt.savefig(os.path.join(PATH, 'value_iter_reward_2.pdf'))
    plt.close(fig)
    visualize_gridworld(world,walk_with_policy(world, policy),"value_iter_reward_2")

    world.reward_class = gw.SecondReward
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    policy,V = value_iteration(world,discount=0.9)
    sb.heatmap(V,annot=True)
    plt.title('Optimal value function (second reward, discount rate=0.9)')
    plt.savefig(os.path.join(PATH, 'value_iter_reward_2_discount_09.pdf'))
    plt.close(fig)
    visualize_gridworld(world,walk_with_policy(world, policy),"value_iter_reward_2")

    world.reward_class = gw.ThirdReward
    fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    policy,V = value_iteration(world)
    sb.heatmap(V,annot=True)
    plt.title('Optimal value function (third reward)')
    plt.savefig(os.path.join(PATH, 'value_iter_reward_3.pdf'))
    plt.close(fig)
    visualize_gridworld(world,walk_with_policy(world, policy),"value_iter_reward_3")
