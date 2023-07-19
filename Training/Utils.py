from collections import namedtuple
from random import random, randrange

import numpy as np
import torch
from torch import device

from Environment.StockEnvironment import ReplayMemory, epsilon_decay
from Models.DQN_Agent import DQN
import torch.optim as optim

Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward',
                         'next_hidden_state1', 'next_hidden_state2'))

device = device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize(architecture, hidden_size, dense_size, dense_layers):
    """
    Initialize environment, DQN networks, optimizer and memory replay.
    """
    feature_size = 32
    num_actions = 11
    dropout_rate = 0.2

    memoryReplay = ReplayMemory(100000)
    Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                    architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                    dropout_rate=dropout_rate)
    target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                         architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                         dropout_rate=dropout_rate)
    target_network.load_state_dict(Q_network.state_dict())
    optimizer = optim.Adam(Q_network.parameters())

    hidden_state1, hidden_state2 = Q_network.init_hidden(1)

    return memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2


def execute_action(state, hidden_state1, hidden_state2, epsilon, num_actions, Q_network):
    sample = random()

    if sample > epsilon:
        with torch.no_grad():
            action, hidden_state1, hidden_state2 = Q_network(state, hidden_state1, hidden_state2)
            action = action.max(1)[1].view(1, 1)
            return action, hidden_state1, hidden_state2, epsilon
    else:
        action = torch.tensor([[randrange(num_actions)]], device=device, dtype=torch.long)
        return action, hidden_state1, hidden_state2, epsilon
