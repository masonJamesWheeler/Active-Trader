import numpy as np           # Handle matrices
import matplotlib.pyplot as plt # Display graphs

from collections import deque

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from StockEnvironment import StockEnvironment, ReplayMemory, EpsilonGreedyStrategy
from data import get_and_process_data
from StockEnvironment import StockEnvironment
import random
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

# Getting Data from AlphaVantage
AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
tickers = ["AAPL"]
interval = '1min'
threshhold = 0.01
window_size = 30
years = 2
months = 12

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, num_features, num_actions, stack_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(stack_size, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(window_size)))
        convf = conv2d_size_out(conv2d_size_out(conv2d_size_out(num_features)))

        linear_input_size = convw * convf * 64
        self.head = nn.Linear(linear_input_size, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

# Data is shaped [num_windows, window_size, num_features]
data = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
print(data[0])

def initialize():
    """
    Initialize environment, DQN networks, optimizer and memory replay.
    """
    starting_cash = 10000
    starting_shares = 100
    window_size = 30
    stack_size = 4
    price_column = 3
    feature_size = 20
    last_state = torch.zeros((1, stack_size, window_size, feature_size), dtype=torch.float32).to('cpu')

    env = StockEnvironment(starting_cash, starting_shares, data, window_size, feature_size, stack_size, price_column)
    memoryReplay = ReplayMemory(100000)
    num_actions = 11
    Q_network = DQN(num_features=feature_size, num_actions= num_actions, stack_size= stack_size)
    target_network = DQN(num_features=feature_size, num_actions= num_actions, stack_size= stack_size)
    target_network.load_state_dict(Q_network.state_dict())
    optimizer = optim.Adam(Q_network.parameters())

    return env, memoryReplay, num_actions, Q_network, target_network, optimizer


def epsilon_decay(steps_done, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    """
    Calculate decay of epsilon using the formula epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
    """
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)


def execute_action(state, steps_done, num_actions, Q_network):
    """
    Execute action based on epsilon-greedy policy.
    """
    if np.random.rand() < epsilon_decay(steps_done):
        action = np.random.randint(num_actions)
    else:
        with torch.no_grad():
            if len(state.shape) == 3:
                state = torch.tensor(state).unsqueeze(0)
            Q_values = Q_network(state)
            action = torch.argmax(Q_values).item()
    return action


def update_Q_values(batch, Q_network, target_network, optimizer, gamma=0.99):
    """
    Update Q values and perform a backward pass.
    """
    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    current_Q_values = Q_network(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()
    next_Q_values = target_network(next_state_batch).max(1)[0].detach()
    target_Q_values = reward_batch + (gamma * next_Q_values)
    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main_loop(num_episodes=1000, C=10, BATCH_SIZE=128):
    """
    Run the main loop of DQN training.
    """
    env, memoryReplay, num_actions, Q_network, target_network, optimizer = initialize()
    steps_done = 0
    for episode in range(num_episodes):
        state = env.reset()
        if len(state.shape) == 3:
            state = torch.tensor(state).unsqueeze(0)
        done = False
        while not done:
            steps_done += 1
            action = execute_action(state, steps_done, num_actions, Q_network)
            next_state, reward, done = env.step(action)
            env.render()

            if len(next_state.shape) == 3:
                next_state = torch.tensor(next_state).unsqueeze(0)

            memoryReplay.push((state, action, next_state, reward))

            if len(memoryReplay) >= BATCH_SIZE:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer)

            if steps_done % C == 0:
                target_network.load_state_dict(Q_network.state_dict())

            state = next_state


if __name__ == "__main__":
    main_loop()

