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
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward', 'next_hidden_state1', 'next_hidden_state2'))


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hidden_state1, hidden_state2):
        x, hidden_state1 = self.gru1(x, hidden_state1)
        x, hidden_state2 = self.gru2(x, hidden_state2)
        x = self.fc(hidden_state2[-1])  # Using the last hidden state of the second GRU
        return x, hidden_state1, hidden_state2

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)

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
    price_column = 3
    feature_size = 20
    window_size = 30
    hidden_size = 128
    num_actions = 11
    #Data size is [1, stack_size, window_size, feature_size]

    env = StockEnvironment(starting_cash, starting_shares, data, window_size, feature_size, price_column)
    memoryReplay = ReplayMemory(100000)
    Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions)
    target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions)
    target_network.load_state_dict(Q_network.state_dict())
    optimizer = optim.Adam(Q_network.parameters())

    hidden_state1, hidden_state2 = Q_network.init_hidden(1)

    return env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2


def epsilon_decay(steps_done, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    """
    Calculate decay of epsilon using the formula epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
    """
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)

def execute_action(state, hidden_state1, hidden_state2, steps_done, num_actions, Q_network):
    """
    Execute action based on epsilon-greedy policy.
    """
    if np.random.rand() < epsilon_decay(steps_done):
        action = np.random.randint(num_actions)
    else:
        with torch.no_grad():
            Q_values, hidden_state1, hidden_state2 = Q_network(state, hidden_state1, hidden_state2)
            action = torch.argmax(Q_values).item()
    return action, hidden_state1, hidden_state2

def update_Q_values(batch, Q_network, target_network, optimizer, gamma=0.99):
    """
    Update Q values and perform a backward pass.
    """
    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    hidden_state1_batch = torch.stack(batch.hidden_state1)
    hidden_state2_batch = torch.stack(batch.hidden_state2)
    next_hidden_state1_batch = torch.stack(batch.next_hidden_state1)
    next_hidden_state2_batch = torch.stack(batch.next_hidden_state2)

    print(state_batch.shape)
    current_Q_values, _, _ = Q_network(state_batch, hidden_state1_batch, hidden_state2_batch)
    current_Q_values = current_Q_values.gather(1, action_batch.view(-1, 1)).squeeze()
    
    next_Q_values, _, _ = target_network(next_state_batch, next_hidden_state1_batch, next_hidden_state2_batch)
    next_Q_values = next_Q_values.max(1)[0].detach()
    
    target_Q_values = reward_batch + (gamma * next_Q_values)
    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def main_loop(num_episodes=1000, C=10, BATCH_SIZE=128):
    """
    Run the main loop of DQN training.
    """
    env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2 = initialize()
    steps_done = 0
    for episode in range(num_episodes):
        state = env.reset()
        print(state.shape)
        hidden_state1, hidden_state2 = Q_network.init_hidden(1)
        done = False
        while not done:
            steps_done += 1
            action, hidden_state1, hidden_state2 = execute_action(state, hidden_state1, hidden_state2, steps_done, num_actions, Q_network)
            next_state, reward, done = env.step(action)
            print(next_state.shape)
            env.render()
            memoryReplay.push((state, hidden_state1, hidden_state2, action, next_state, reward, hidden_state1, hidden_state2))

            if len(memoryReplay) >= BATCH_SIZE:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer)

            if steps_done % C == 0:
                target_network.load_state_dict(Q_network.state_dict())

            state = next_state



if __name__ == "__main__":
    main_loop()

