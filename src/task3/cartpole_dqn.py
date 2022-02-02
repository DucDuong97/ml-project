import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from matplotlib import pyplot as plt
from torch.nn.modules.loss import MSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')

# Hyper-parameters
num_ep = 1500
prob = 1.0
min_prob = 0.02
discount_rate = 0.99
decay_rate = 0.99
lr = 1e-3


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


def train(train_model, target_model, transition, discount_rate, optimizer, loss_func=MSELoss()):
    state, action, reward, next_state, done = transition[0], transition[1], \
                                              transition[2], transition[3], transition[4]
    state, next_state, done = torch.tensor(state, device=device), \
                        torch.tensor(next_state, device=device), torch.tensor(done, device=device)

    q = train_model(state).gather(1, torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(-1))

    with torch.no_grad():
        max_arg = target_model(next_state).detach().max(1)[0]
        q_target = torch.tensor(reward, dtype=torch.float, device=device)
        q_target[~done] += discount_rate * max_arg[~done]


    optimizer.zero_grad()
    loss = loss_func(q, q_target.unsqueeze(1))
    loss.backward()
    optimizer.step()

    return loss.item()


train_model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(train_model.state_dict())

optimizer = optim.Adam(train_model.parameters(), lr=lr)

print(f'Run DQN without experience replay')

episodes = []
rewards = []

for ep in range(num_ep):
    state = env.reset()
    total_reward = 0.0
    done = False

    prob = max(min_prob, prob * decay_rate)

    while not done:
        if np.random.rand(1) < prob:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                output = train_model(torch.tensor(state, dtype=torch.float, device=device)).cpu().numpy()
            action = np.argmax(output)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        transition = (np.array([state]), np.array([action]),
                      np.array([reward]), np.array([next_state]), np.array(done))

        target_model.load_state_dict(train_model.state_dict())
        loss = train(train_model=train_model, target_model=target_model, transition=transition,
                     discount_rate=discount_rate, optimizer=optimizer)
        state = next_state

        episodes.append(ep)
        rewards.append(total_reward)
    print(f'Episode {ep} - Total Reward:  {total_reward}')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(episodes, rewards, label="Reward")
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Reward each episode', fontsize=14)
plt.show()