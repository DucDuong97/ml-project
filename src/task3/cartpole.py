import math
import random
import gym
from pyglet.window import key
import numpy as np
import matplotlib.pyplot as plt


def interactive_cartpole():
    """
    Allows you to control the cart with the arrow keys.
    Press Space to reset the cart-pole and press Escape to close the window.
    """

    env = gym.make('CartPole-v1')
    # Make sure to render once so that the viewer exists
    env.reset()
    env.render()
    # Inject key_handler
    key_handler = key.KeyStateHandler()
    env.viewer.window.push_handlers(key_handler)

    action = 0
    already_done = False
    t = 0
    while True:
        if key_handler[key.ESCAPE]:
            break

        if key_handler[key.SPACE]:
            env.reset()
            action = 0
            t = 0
            already_done = False

        if key_handler[key.LEFT]:
            action = 0
        elif key_handler[key.RIGHT]:
            action = 1

        observation, reward, done, info = env.step(action)
        env.render()

        if not done:
            print(observation, reward, done, info)

        if done and not already_done:
            print(f'Episode finished after {t + 1} time steps')
            already_done = True

        t += 1

    env.close()

num_bucket_x = 48
size_bucket_x = 0.1
x_lower_bound = -2.4
x_upper_bound = 2.4

num_bucket_x_vec = 48*2
size_bucket_x_vec = 0.1
x_vec_lower_bound = -4.8
x_vec_upper_bound = 4.8

num_bucket_theta = 8
theta_lower_bound = -12 * 2 * math.pi / 360
theta_upper_bound = 12 * 2 * math.pi / 360
size_bucket_theta = (theta_upper_bound - theta_lower_bound)/num_bucket_theta

num_bucket_theta_vec = 8
theta_vec_lower_bound = -12 * 20 * 2 * math.pi / 360
theta_vec_upper_bound = 12 * 20 * 2 * math.pi / 360
size_bucket_theta_vec = (theta_vec_upper_bound - theta_vec_lower_bound)/num_bucket_theta_vec


def discret_x(value):
    return math.floor((value - x_lower_bound) / size_bucket_x)

def discret_x_vec(value):
    return math.floor((value - x_vec_lower_bound) / size_bucket_x_vec)

def discret_theta(value):
    return math.floor((value - theta_lower_bound) / size_bucket_theta)

def discret_theta_vec(value):
    return math.floor((value - theta_vec_lower_bound) / size_bucket_theta_vec)


def q_learn_cartpole(env, eps_num=200, lr=0.9, max_step=500):

    Q = np.zeros((num_bucket_theta, num_bucket_theta_vec, 2))
    high_score = 0
    best_moves = []
    eps_scores = []

    for ep in range(eps_num):
        last_scores = eps_scores[-5:]
        last_scores_avg = sum(last_scores) / max(1,len(last_scores))
        er = max(0,1 - last_scores_avg/40)
        print('-----------------------------')
        print(f"EXECUTING EPISODE {ep}, er = {er}")
        moves = []
        done = False

        env.reset()
        env.render()
        S = env.state
        for step in range(max_step):
            # Step 1
            A = choose_action(env, Q, er)
            # Step 2
            new_state, reward, done, info = env.step(A)
            env.render()
            if done or step == max_step - 1:
                eps_scores.append(step)
                if step > high_score:
                    high_score = step
                    best_moves = moves
                break
            # Step 3: Update Q
            lookup_values = Q[discret_theta(new_state[2]),discret_theta_vec(new_state[3]), :]
            Q[discret_theta(S[2]),discret_theta_vec(S[3]), A] += lr * (reward + 
                        max(lookup_values) - Q[discret_theta(S[2]),discret_theta_vec(S[3]), A])
            # Retriving observed data and print info
            moves.append(A)
            # Step 4
            S = new_state
    print(f'High score: {high_score}')
    print(f'Best moves: {best_moves}')
    plot_ret_eps(eps_scores)
    return Q


def plot_ret_eps(returns):
    plt.scatter(range(len(returns)),returns)
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.title('Return on Episode')
    plt.show()


def choose_action(env, Q, prob):
    choice = random.choices(["random_action", "argmax_action"], [prob, 1 - prob], k=1)[0]
    if choice == "random_action":
        return env.action_space.sample()
    else:
        lookup_values = Q[discret_theta(env.state[2]),discret_theta_vec(env.state[3]), :]
        return np.argmax(lookup_values)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    q_learn_cartpole(env)
