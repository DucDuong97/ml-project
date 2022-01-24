import math
import random
import gym
from pyglet.window import key
import numpy as np

env = gym.make('CartPole-v1')


def interactive_cartpole():
    """
    Allows you to control the cart with the arrow keys.
    Press Space to reset the cart-pole and press Escape to close the window.
    """

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


num_bucket_x = 3
num_bucket_delta_x = 3
num_bucket_theta = 12
num_bucket_delta_theta = 6
value_bounds = list(zip(env.observation_space.low, env.observation_space.high))


def q_learn_cartpole(num_ep, prob, step_size=1, max_step=500):
    Q = np.zeros((num_bucket_x, num_bucket_delta_x, num_bucket_theta, num_bucket_delta_theta, 2))
    returns = []
    moves = []

    for i in range(num_ep):
        print('-----------------------------')
        print(f"EXECUTING EPISODE {i}")
        done = False

        env.reset()
        env.render()
        S = env.state
        current_step = 0
        moves = []
        ret = 0
        while not done and current_step < max_step:
            # Step 1
            A = choose_action(env, Q, prob)
            # Step 2
            new_state, reward, done, info = env.step(A)

            if done:
                print(current_step)
                break

            # Step 3: Update Q
            lookup_values = Q[  map_to_bucket(new_state[0], value_bounds[0], num_bucket_x),
                                map_to_bucket(new_state[1], value_bounds[1], num_bucket_delta_x),
                                map_to_bucket(new_state[2], value_bounds[2], num_bucket_theta),
                                map_to_bucket(new_state[3], value_bounds[3], num_bucket_delta_theta), :]
            max_arg = max(lookup_values)
            Q[map_to_bucket(S[0], value_bounds[0], num_bucket_x),
              map_to_bucket(S[1], value_bounds[1], num_bucket_delta_x),
              map_to_bucket(S[2], value_bounds[2], num_bucket_theta),
              map_to_bucket(S[3], value_bounds[3], num_bucket_delta_theta), A] \
                += step_size * (reward + max_arg - Q[map_to_bucket(S[0], value_bounds[0], num_bucket_x),
                                                     map_to_bucket(S[1], value_bounds[1], num_bucket_delta_x),
                                                     map_to_bucket(S[2], value_bounds[2], num_bucket_theta),
                                                     map_to_bucket(S[3], value_bounds[3], num_bucket_delta_theta), A])
            # Retriving observed data and print info
            ret += reward
            moves.append(A)
            current_step += 1
            # Step 4
            S = new_state
        env.close()
    return Q


def choose_action(env, Q, prob):
    choice = random.choices(["random_action", "argmax_action"], [prob, 1 - prob], k=1)[0]
    if choice == "random_action":
        return env.action_space.sample()
    else:
        lookup_values = Q[  map_to_bucket(env.state[0], value_bounds[0], num_bucket_x),
                            map_to_bucket(env.state[1], value_bounds[1], num_bucket_delta_x),
                            map_to_bucket(env.state[2], value_bounds[2], num_bucket_theta),
                            map_to_bucket(env.state[3], value_bounds[3], num_bucket_delta_theta), :]
        max_lookup_value = max(lookup_values)
        return list(lookup_values).index(max_lookup_value)


def map_to_bucket(value, bounds, num_bucket):
    bucket_size = (bounds[1] - bounds[0]) / num_bucket
    if value == bounds[0]:
        return 0
    elif value == bounds[1]:
        num_bucket - 1
    return math.floor(value / bucket_size)


if __name__ == '__main__':
    print(q_learn_cartpole(800, 0.01))
