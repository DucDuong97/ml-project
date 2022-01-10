import json
import random

import numpy as np


def apply_action(state, action):
    x, y = state.x, state.y
    if action == 'left':
        x -= 1
    elif action == 'right':
        x += 1
    elif action == 'up':
        y -= 1
    elif action == 'down':
        y += 1

    return x, y


class Cell:
    """
    Cell base class. This corresponds to the behaviour of a blank cell
    """
    def __init__(self, world, x, y):
        self.world = world
        self.x = x
        self.y = y
        self.reachable = True
        self.terminal = False

    def to_dict(self):
        res = self.__dict__.copy()
        res['__cls__'] = type(self).__name__
        del res['world']
        return res

    def from_dict(self, dictionary):
        if '__cls__' in dictionary:
            del dictionary['__cls__']

        self.__dict__.update(dictionary)

    def step(self, action):
        """
        Sample the next state when an agent performs the action while in this state.

        :param action: The action that the agent wants to take
        :return: the resulting state
        """
        if self.terminal:
            return self

        x, y = apply_action(self, action)

        state = self.world.get_state(x, y)

        return state

    def allow_enter(self, old_state, action):
        """
        Specify whether the agent can enter this state. The base implementation simply checks if the old state is next
        to the current state.
        :param old_state: The previous state that the agent is coming from
        :param action: The action that the agent wants to take
        :return: bool that specifies if the agent can enter or not
        """
        if not self.reachable:
            return False

        return (abs(self.x - old_state.x) + abs(self.y - old_state.y)) <= 1

    def get_afterstates(self, action):
        """
        This should return all possible states that can be output by step
        if that method is called with parameter action
        """
        if self.terminal:
            return [self]

        x, y = apply_action(self, action)
        return [self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        """
        Compute the probability that self.step(action) will return new_state
        """
        if self.terminal:
            return int(new_state == self)

        x, y = apply_action(self, action)

        return int(new_state == self.world.get_state(x, y))

    def p_enter(self, old_state, action):
        """
        Compute the probability that self.allow_enter(old_state, action) will return True
        """
        if self.allow_enter(old_state, action):
            return 1

        return 0

    def __eq__(self, other):
        """
        Overwriting __eq__ and __hash__ means that Cells can be used as dictionary keys.
        """
        if not isinstance(other, Cell):
            return False

        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """
        Overwriting __eq__ and __hash__ means that Cells can be used as dictionary keys.
        """
        return hash((self.x, self.y))

    def __str__(self):
        return f'{type(self).__name__} at ({self.x}, {self.y})'

    def __repr__(self):
        return str(self)


class BlankCell(Cell):
    pass

class StartCell(Cell):
    def allow_enter(self, old_state, action):
        return super(StartCell, self).allow_enter(old_state, action) or old_state.terminal

class GoalCell(Cell):
    def __init__(self, world, x, y):
        super(GoalCell, self).__init__(world, x, y)

        self.terminal = True

class WallCell(Cell):
    def __init__(self, world, x, y):
        super(WallCell, self).__init__(world, x, y)

        self.reachable = False

class ArrowCell(Cell):
    def __init__(self, world, x, y, direction='left'):
        super(ArrowCell, self).__init__(world, x, y)

        self.direction = direction
    
    def step(self, action):
        return Cell.step(self, self.direction)

    def get_afterstates(self, action):
        x, y = apply_action(self, self.direction)
        return [self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        if new_state == self.step(action):
            return 1

        return 0

class SwampCell(Cell):
    def __init__(self, world, x, y, stick_prob=0.5):
        super(SwampCell, self).__init__(world, x, y)

        self.stick_prob = stick_prob
    
    def step(self, action):
        if self.world.rng.random() < self.stick_prob:
            return self
        
        return super(SwampCell, self).step(action)

    def get_afterstates(self, action):
        x, y = apply_action(self, action)
        return [self, self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        if new_state == Cell.step(self, action):
            return 1 - self.stick_prob
        elif new_state == self:
            return self.stick_prob

        return 0

class PitCell(GoalCell):
    pass
        

class DefaultReward:
    possible_rewards = [0, -1, -1000]

    @staticmethod
    def reward_f(old_state, action, new_state):
        if old_state.terminal:
            return 0
        if isinstance(new_state, PitCell):
            return -1000

        return -1

    @staticmethod
    def reward_p(reward, new_state, old_state, action):
        """
        Computes p(R_{t+1} | S_{t+1}=new_state, S_t=old_state, A_t=action)
        """
        if old_state.terminal:
            true_r = 0
        elif isinstance(new_state, PitCell):
            true_r = -1000
        else:
            true_r = -1

        return int(true_r == reward)


class World:
    def __init__(self, reward_class=DefaultReward):
        self.reward_class = reward_class
        self.rng = random.Random()
        self.grid = None
        self.start_states = None
        self.current_state = None

    def save_to_file(self, path):
        g_list = self.grid.flatten().tolist()
        g_list = [o for o in g_list if not isinstance(o, BlankCell)]

        world_info = dict(size=self.grid.shape, reward_class=self.reward_class.__name__,
                          grid=g_list)

        class CellEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()

                return json.JSONEncoder.default(self, obj)

        with open(path, mode='w') as f:
            json.dump(world_info, f, cls=CellEncoder, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_file(cls, path):
        def hook(dct):
            if '__cls__' not in dct:
                return dct

            klass = globals()[dct['__cls__']]
            obj = klass(None, dct['x'], dct['y'])
            obj.from_dict(dct)
            return obj

        with open(path, mode='r') as f:
            world_info = json.load(f, object_hook=hook)

        rew_class = globals()[world_info['reward_class']]
        world = World(rew_class)
        world.start_states = []

        world.grid = np.array([[BlankCell(world, x, y) for x in range(world_info['size'][0])]
                              for y in range(world_info['size'][1])], dtype=np.object)
        for cell in world_info['grid']:
            cell.world = world
            if isinstance(cell, StartCell):
                world.start_states.append(cell)
            world.grid[cell.y, cell.x] = cell

        return world

    def get_state(self, x, y):
        if self.grid is None:
            raise RuntimeError
        if not (0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]):
            raise ValueError

        return self.grid[y, x]

    def reset(self):
        self.current_state = self.rng.choice(self.start_states)

    def step(self, action):
        proposed_state = self.current_state.step(action)
        if proposed_state.allow_enter(self.current_state, action):
            new_state = proposed_state
        else:
            new_state = self.current_state

        reward = self.reward_class.reward_f(self.current_state, action, new_state)
        done = new_state.terminal
        self.current_state = new_state

        return new_state, reward, done

    def p(self, new_state, reward, old_state, action):
        """
        Computes p(S_{t+1}=new_state, R_{t+1}=reward | S_t=old_state, A_t=action)
        """
        reward_p = self.reward_class.reward_p(reward, new_state, old_state, action)
        if reward_p == 0:
            return 0

        step_p = old_state.p_step(action, new_state)

        if new_state != old_state:
            enter_p = new_state.p_enter(old_state, action)
            return reward_p * step_p * enter_p

        if step_p == 1:
            return reward_p

        sum = 0
        count = 0
        for s in old_state.get_afterstates(action):
            if s != old_state:
                sum += s.p_enter(old_state, action)
                count += 1
        return reward_p * (step_p + (1 - step_p)*(1 - sum/count))


def random_walk(steps=100):
    world = World.load_from_file('world.json')
    world.reset()

    print(f'Starting at pos. ({world.current_state.x}, {world.current_state.y}).')

    ep_return = 0
    last_termination = 0
    for s in range(steps):
        print(f'Step {s+1}/{steps}...')
        action = random.choice(['left', 'right', 'up', 'down'])
        print(f'Going {action}!')
        new_state, reward, done = world.step(action)
        ep_return += reward
        print(f'Received a reward of {reward}!')
        print(f'New position is ({new_state.x}, {new_state.y}) on a {type(new_state).__name__}.')
        if done:
            print(f'Episode terminated after {s+1-last_termination} steps. Total Return was {ep_return}.')
            ep_return = 0
            last_termination = s+1
            print(f'Resetting the world...')
            world.reset()


if __name__ == '__main__':
    random_walk(1000)
