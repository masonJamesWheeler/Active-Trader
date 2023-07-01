import base64
import collections
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from collections import namedtuple

from StockEnvironment import EpsilonGreedyStrategy, ReplayMemory, StockEnvironment
from data import get_and_process_data

warnings.filterwarnings('ignore')

Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward', 'next_hidden_state1', 'next_hidden_state2'))


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, architecture, dense_layers, dense_size, dropout_rate):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size

        # Choose architecture based on the input argument
        if architecture == 'RNN':
            RNN_Layer = nn.RNN
        elif architecture == 'GRU':
            RNN_Layer = nn.GRU
        elif architecture == 'LSTM':
            RNN_Layer = nn.LSTM
        else:
            raise ValueError(f'Invalid architecture: {architecture}')

        self.rnn1 = RNN_Layer(input_size, hidden_size, batch_first=True)
        self.rnn2 = RNN_Layer(hidden_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after RNN layer 1
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after RNN layer 2

        # Dynamic construction of dense layers
        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Dropout after each dense layer
        for i in range(dense_layers):
            if i == 0:
                self.dense_layers.append(nn.Linear(hidden_size, dense_size))
            else:
                self.dense_layers.append(nn.Linear(dense_size // (2 ** (i-1)), dense_size // (2 ** i)))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(dense_size // (2 ** (dense_layers-1)), num_actions)

    def forward(self, x, hidden_state1, hidden_state2):
        if isinstance(self.rnn1, nn.LSTM):
            h1, c1 = hidden_state1
            h2, c2 = hidden_state2
            x, (h1, c1) = self.rnn1(x, (h1, c1))
            x = self.dropout1(x)  # Apply dropout
            x, (h2, c2) = self.rnn2(x, (h2, c2))
            x = self.dropout2(x)  # Apply dropout
            hidden_state1 = (h1, c1)
            hidden_state2 = (h2, c2)
        else:
            x, hidden_state1 = self.rnn1(x, hidden_state1)
            x = self.dropout1(x)  # Apply dropout
            x, hidden_state2 = self.rnn2(x, hidden_state2)
            x = self.dropout2(x)  # Apply dropout

        x = hidden_state2[-1]  # Using the last hidden state of the second GRU

        # Pass through dynamic dense layers
        for i, (dense_layer, dropout_layer) in enumerate(zip(self.dense_layers, self.dropout_layers)):
            x = F.relu(dense_layer(x))
            x = dropout_layer(x)  # Apply dropout

        x = self.output_layer(x)

        return x, hidden_state1, hidden_state2

    def init_hidden(self, batch_size):
        if isinstance(self.rnn1, nn.LSTM):
            return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)), \
                (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
        else:
            return torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)
    
def initialize(data, scaled_data, architecture, window_size, hidden_size, dense_size, dense_layers, reward_function):
    """
    Initialize environment, DQN networks, optimizer and memory replay.
    """
    starting_cash = 10000
    starting_shares = 100
    window_size = 128
    price_column = 3
    feature_size = 30
    hidden_size = 128
    num_actions = 11
    dropout_rate = 0.2
    #Data size is [1, stack_size, window_size, feature_size]

    env = StockEnvironment(starting_cash, starting_shares, data, scaled_data, window_size, feature_size, price_column, reward_function=reward_function)
    memoryReplay = ReplayMemory(20000)
    Q_network = DQN(input_size=feature_size+2, hidden_size=hidden_size, num_actions=num_actions, architecture=architecture, dense_layers=dense_layers, dense_size=dense_size, dropout_rate = dropout_rate)
    target_network = DQN(input_size=feature_size+2, hidden_size=hidden_size, num_actions=num_actions, architecture=architecture, dense_layers=dense_layers, dense_size=dense_size, dropout_rate = dropout_rate)
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

def update_Q_values(batch, Q_network, target_network, optimizer, architecture, gamma=0.99):
    """
    Update the Q values and perform a backward pass.

    Args:
        batch: A batch of experiences containing states, actions, rewards, next states and hidden states.
        Q_network: The current Q network model.
        target_network: The target Q network model.
        optimizer: The optimizer used to update the weights of the Q network.
        architecture (str): The architecture of the Q network (e.g., 'LSTM').
        gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.

    Returns:
        None
    """
    # Convert the batch data into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    def process_batch(hidden_state):
        """
        Process the hidden states from the batch.
        
        Args:
            hidden_state: A list of hidden states.

        Returns:
            A tensor containing the processed hidden states.
        """
        # Check if hidden state is a tuple (for LSTM)
        if isinstance(hidden_state[0], tuple):
            return (torch.stack([x[0] for x in hidden_state]).squeeze().unsqueeze(0), torch.stack([x[1] for x in hidden_state]).squeeze().unsqueeze(0))
        else:
            return torch.stack(hidden_state).squeeze().unsqueeze(0)

    # Process the hidden states
    hidden_state1_batch = process_batch(batch.hidden_state1)
    hidden_state2_batch = process_batch(batch.hidden_state2)
    next_hidden_state1_batch = process_batch(batch.next_hidden_state1)
    next_hidden_state2_batch = process_batch(batch.next_hidden_state2)

    # Compute current Q values
    current_Q_values, _, _ = Q_network(state_batch, hidden_state1_batch, hidden_state2_batch)

    # For LSTM architecture, remove unnecessary dimension
    if architecture == 'LSTM':
        current_Q_values = current_Q_values.squeeze()
    # Gather the Q values for the actions taken
    current_Q_values = current_Q_values.gather(1, action_batch.view(-1, 1)).squeeze()

    # Compute Q values for next states
    next_Q_values, _, _ = target_network(next_state_batch, next_hidden_state1_batch, next_hidden_state2_batch)
    # For LSTM architecture, remove unnecessary dimension
    if architecture == 'LSTM':
        next_Q_values = next_Q_values.squeeze()
    # Detach the Q values and get the maximum Q value for each next state
    next_Q_values = next_Q_values.max(1)[0].detach()

    # Compute the target Q values
    target_Q_values = reward_batch + (gamma * next_Q_values)
    # Compute the loss between current and target Q values
    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

    # Zero the gradients
    optimizer.zero_grad()
    # Perform a backward pass to compute gradients
    loss.backward()
    # Update the weights of the Q network
    optimizer.step()

def main_loop(tickers, num_episodes=100, C=10, BATCH_SIZE=524, architecture='RNN', window_size=128, hidden_size=128, dense_size=128, dense_layers=2, reward_function='linear'):
    """
    Run the main loop of DQN training.
    """
    # Set the interval, threshold, years, and months for data retrieval
    interval = '1min'
    threshhold = 0.01
    years = 2
    months = 12
    # Set the AlphaVantage API key
    AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
    # Retrieve and process data for the first ticker
    data, scaled_data = get_and_process_data(tickers[0], interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)

    # Initialize the environment, memory replay, Q-network, target network, optimizer, and hidden states
    env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2 = initialize(architecture=architecture, data=data, scaled_data = scaled_data, window_size=window_size, hidden_size=hidden_size, dense_size=dense_size, dense_layers=dense_layers, reward_function=reward_function)
    steps_done = 0
    iterator = 0
    # Loop through all tickers
    for ticker in range(len(tickers)):
        iterator += 1
        # Retrieve and process data for the current ticker
        env.data, env.scaled_data = get_and_process_data(tickers[ticker], interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
        # Reset the environment and initialize the hidden states
        state = env.reset()
        hidden_state1, hidden_state2 = Q_network.init_hidden(1)
        done = False
        # Loop until the episode is done
        while not done:
            steps_done += 1
            # Execute an action and get the next state, reward, and done flag
            action, hidden_state1, hidden_state2 = execute_action(state, hidden_state1, hidden_state2, steps_done, num_actions, Q_network)
            next_state, reward, done = env.step(action)
            # Render the environment
            # Add the transition to the memory replay
            memoryReplay.push((state, hidden_state1, hidden_state2, action, next_state, reward, hidden_state1, hidden_state2))

            # If the memory replay is full, sample a batch and update the Q-values
            if len(memoryReplay) >= BATCH_SIZE:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer, architecture)

            # If the number of steps is a multiple of C, update the target network
            if steps_done % C == 0:
                target_network.load_state_dict(Q_network.state_dict())

            state = next_state

        torch.save(Q_network.state_dict(), f'Q_network_weights_episode_{iterator}.pth')


if __name__ == "__main__":
    tickers = ["MSFT", "META", "TSLA", "NVDA", "PYPL", "ADBE", "NFLX"]

    main_loop(tickers = tickers)

