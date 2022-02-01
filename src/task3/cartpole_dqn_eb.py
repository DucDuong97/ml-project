import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
import gym
from torch.nn.modules.loss import MSELoss
#from cartpole_dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')

# Hyper-parameters
num_ep = 10000
batch_size = 32
buffer_size = 1000000
prob = 1.0
min_prob = 0.02
discount_rate = 0.99
decay_rate = 0.99
lr = 1e-4
update_rate = 100


class DQN(nn.Module):
    def __init__(self, input_layer=env.observation_space.shape[0],
                 output_layer=env.action_space.n):
        super(DQN, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.fc1 = nn.Linear(input_layer, 32)
        self.fc2 = nn.Linear(32, output_layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ExperienceReplay:
    def __init__(self, batch_size, buffer_size):
        self.buffer = collections.deque()
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def push(self, *transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def get_length(self):
        return len(self.buffer)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[index] for index in indices]

        state, action, reward, next_state, done = zip(*batch)

        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        return state, action, reward, next_state, done


def train(train_model, target_model, discount_rate, optimizer, buffer, loss_func=MSELoss()):
    state, action, reward, next_state, _ = buffer.sample()
    state, next_state = torch.tensor(state, device=device), \
                        torch.tensor(next_state, device=device)

    q = train_model(state).gather(1, torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(-1))

    with torch.no_grad():
        max_arg = target_model(next_state).detach().max(1)[0]
        q_target = torch.tensor(reward, dtype=torch.float, device=device) + discount_rate * max_arg

    optimizer.zero_grad()
    loss = loss_func(q, q_target.unsqueeze(1))
    loss.backward()
    optimizer.step()

    return loss.item()


train_model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(train_model.state_dict())

optimizer = optim.Adam(train_model.parameters(), lr=lr)
exp_buffer = ExperienceReplay(batch_size, buffer_size)

while exp_buffer.get_length() < buffer_size:
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        exp_buffer.push(state, action, reward, next_state, done)
        state = next_state

update_counter = 0

print(f'Run DQN with experience replay')

for ep in range(num_ep):
    state = env.reset()
    total_reward = 0.0
    total_loss = 0.0
    done = False

    prob = max(min_prob, prob * decay_rate)

    while not done:
        update_counter += 1
        if np.random.rand(1) < prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                output = train_model(torch.tensor(state, dtype=torch.float, device=device)).cpu().numpy()
            action = np.argmax(output)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        exp_buffer.push(state, action, reward, next_state, done)

        if update_counter == update_rate:
            update_counter = 0
            target_model.load_state_dict(train_model.state_dict())
            loss = train(train_model=train_model, target_model=target_model, discount_rate=discount_rate,
                         optimizer=optimizer, buffer=exp_buffer)
            total_loss += loss

        state = next_state
    print(f'Episode {ep} - Total Reward:  {total_reward}')
