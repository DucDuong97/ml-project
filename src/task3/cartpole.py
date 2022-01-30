import math
import random
import gym
from pyglet.window import key
import numpy as np



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


num_bucket_theta = 24
size_bucket_theta = 1
theta_lower_bound = -12
theta_upper_bound = 12


num_bucket_theta_vec = 24
size_bucket_theta_vec = 1
theta_vec_lower_bound = -12
theta_vec_upper_bound = 12

def discret_x(value):
    if value == x_lower_bound:
        return 0
    elif value == x_upper_bound:
        return num_bucket_x - 1
    return math.floor(value / size_bucket_x)

def discret_x_vec(value):
    if value == x_vec_lower_bound:
        return 0
    elif value == x_vec_upper_bound:
        return num_bucket_x_vec - 1
    return math.floor(value / size_bucket_x_vec)

def discret_theta(value):
    if value == theta_lower_bound:
        return 0
    elif value == theta_upper_bound:
        return num_bucket_theta - 1
    return math.floor(value / size_bucket_theta)

def discret_theta_vec(value):
    if value == theta_vec_lower_bound:
        return 0
    elif value == theta_vec_upper_bound:
        return num_bucket_theta_vec - 1
    return math.floor(value / size_bucket_theta_vec)


def q_learn_cartpole(env, num_ep, prob, step_size=1, max_step=500):

    Q = np.zeros((num_bucket_x, num_bucket_x_vec, num_bucket_theta, num_bucket_theta_vec, 2))
    moves = []

    for i in range(num_ep):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False

        env.reset()
        #env.render()
        S = env.state
        moves = []
        ret = 0
        for i in range(max_step):
            # Step 1
            A = choose_action(env, Q, prob)
            # Step 2
            new_state, reward, done, info = env.step(A)

            #print(new_state)
            if done:
                print(i)
                break

            # Step 3: Update Q
            lookup_values = Q[discret_x(new_state[0]),discret_x_vec(new_state[1]),
                              discret_theta(new_state[2]),discret_theta_vec(new_state[3]), :]
            Q[discret_x(S[0]),discret_x_vec(S[1]),
            discret_theta(S[2]),discret_theta_vec(S[3]), A] \
                += step_size * (reward + max(lookup_values) - 
                    Q[discret_x(S[0]),discret_x_vec(S[1]),
                    discret_theta(S[2]),discret_theta_vec(S[3]), A])
            # Retriving observed data and print info
            ret += reward
            moves.append(A)
            # Step 4
            S = new_state
    return Q


def choose_action(env, Q, prob):
    choice = random.choices(["random_action", "argmax_action"], [prob, 1 - prob], k=1)[0]
    if choice == "random_action":
        return env.action_space.sample()
    else:
        lookup_values = Q[discret_x(env.state[0]),discret_x_vec(env.state[1]),
                              discret_theta(env.state[2]),discret_theta_vec(env.state[3]), :]
        max_lookup_value = max(lookup_values)
        return list(lookup_values).index(max_lookup_value)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    q_learn_cartpole(env, 800, 0.01)

