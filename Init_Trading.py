import collections
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from collections import namedtuple
from alpaca.trading.client import TradingClient
import alpaca_trade_api as tradeapi
import pytz
from dotenv import load_dotenv
import os
from Environment.LiveStockEnvironment import LiveStockEnvironment
from Environment.StockEnvironment import ReplayMemory
from Models.DQN_Agent import DQN

# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

warnings.filterwarnings('ignore')

Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward',
                         'next_hidden_state1', 'next_hidden_state2'))

api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets',
                         api_version='v2')
trading_client = TradingClient(paper_alpaca_key, paper_alpaca_secret_key)

#
#
# def initialize(architecture, hidden_size, dense_size, dense_layers):
#     """
#     Initialize environment, DQN networks, optimizer and memory replay.
#     """
#     feature_size = 46
#     num_actions = 11
#     dropout_rate = 0.2
#
#     memoryReplay = ReplayMemory(100000)
#     Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
#                     architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
#                     dropout_rate=dropout_rate)
#     target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
#                          architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
#                          dropout_rate=dropout_rate)
#     target_network.load_state_dict(Q_network.state_dict())
#     optimizer = optim.Adam(Q_network.parameters())
#
#     hidden_state1, hidden_state2 = Q_network.init_hidden(1)
#
#     return memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2
#
#
# def epsilon_decay(steps_done, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
#     """
#     Calculate decay of epsilon using the formula epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
#     """
#     return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
#
#
# def execute_action(state, hidden_state1, hidden_state2, steps_done, num_actions, Q_network):
#     """
#     Execute action based on epsilon-greedy policy.
#     """
#     if np.random.rand() < epsilon_decay(steps_done):
#         action = np.random.randint(num_actions)
#     else:
#         with torch.no_grad():
#             Q_values, hidden_state1, hidden_state2 = Q_network(state, hidden_state1, hidden_state2)
#             action = torch.argmax(Q_values).item()
#     return action, hidden_state1, hidden_state2
#
#
# def update_Q_values(batch, Q_network, target_network, optimizer, gamma=0.99):
#     """
#     Update the Q values and perform a backward pass.
#
#     Args:
#         batch: A batch of experiences containing states, actions, rewards, next states and hidden states.
#         Q_network: The current Q network model.
#         target_network: The target Q network model.
#         optimizer: The optimizer used to update the weights of the Q network.
#         architecture (str): The architecture of the Q network (e.g., 'LSTM').
#         gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
#
#     Returns:
#         None
#     """
#     # Convert the batch data into tensors
#     state_batch = torch.cat(batch.state)
#     action_batch = torch.tensor(batch.action)
#     reward_batch = torch.tensor(batch.reward)
#     next_state_batch = torch.cat(batch.next_state)
#
#     def process_batch(hidden_state):
#         """
#         Process the hidden states from the batch.
#
#         Args:
#             hidden_state: A list of hidden states.
#
#         Returns:
#             A tensor containing the processed hidden states.
#         """
#         # Check if hidden state is a tuple (for LSTM)
#         if isinstance(hidden_state[0], tuple):
#             return (torch.stack([x[0] for x in hidden_state]).squeeze().unsqueeze(0),
#                     torch.stack([x[1] for x in hidden_state]).squeeze().unsqueeze(0))
#         else:
#             return torch.stack(hidden_state).squeeze().unsqueeze(0)
#
#     # Process the hidden states
#     hidden_state1_batch = process_batch(batch.hidden_state1)
#     hidden_state2_batch = process_batch(batch.hidden_state2)
#     next_hidden_state1_batch = process_batch(batch.next_hidden_state1)
#     next_hidden_state2_batch = process_batch(batch.next_hidden_state2)
#
#     # Compute current Q values
#     current_Q_values, _, _ = Q_network(state_batch, hidden_state1_batch, hidden_state2_batch)
#
#     # Gather the Q values for the actions taken
#     current_Q_values = current_Q_values.gather(1, action_batch.view(-1, 1)).squeeze()
#
#     # Compute Q values for next states
#     next_Q_values, _, _ = target_network(next_state_batch, next_hidden_state1_batch, next_hidden_state2_batch)
#     # Detach the Q values and get the maximum Q value for each next state
#     next_Q_values = next_Q_values.max(1)[0].detach()
#
#     # Compute the target Q values
#     target_Q_values = reward_batch + (gamma * next_Q_values)
#     # Compute the loss between current and target Q values
#     loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
#
#     # Zero the gradients
#     optimizer.zero_grad()
#     # Perform a backward pass to compute gradients
#     loss.backward()
#     # Update the weights of the Q network
#     optimizer.step()
#
#
# def main_loop(env, BATCH_SIZE=524, architecture='RNN', hidden_size=128,
#               dense_size=128, dense_layers=2):
#     """
#     Run the main loop of DQN training.
#     Args:
#         env: The trading environment.
#         BATCH_SIZE: The size of the batch for training the network.
#         architecture: The architecture of the network.
#         window_size: The size of the window for the network.
#         hidden_size: The size of the hidden layer for the network.
#         dense_size: The size of the dense layer for the network.
#         dense_layers: The number of dense layers in the network.
#         reward_function: The reward function for the network.
#     """
#     # Initialize the environment and set the mode to "slow"
#     live_env = env
#
#     live_env.initialize()
#     live_env.mode = "slow"
#
#     # Set the constants for saving and updating the network
#     C = 10
#     Save_every = 1000
#
#     # Get the initial state of the environment
#     state = live_env.get_state()
#
#     # Initialize the memory replay, Q-network, target network, optimizer, and hidden states
#     memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2 = initialize(
#         architecture=architecture, hidden_size=hidden_size, dense_size=dense_size,
#         dense_layers=dense_layers)
#
#     hidden_state1, hidden_state2 = Q_network.init_hidden(1)
#
#     # If the weights exist, then load them
#     if os.path.exists("Models/{}.pth".format(ticker)):
#             Q_network.load_state_dict(torch.load(f"Models/{ticker}.pth"))
#             target_network.load_state_dict(torch.load(f"Models/{ticker}_target.pth"))
#
#     steps_done = 0
#     # Initialize a deque to store delayed states and actions
#     delay_length = 2
#     delayed_states = collections.deque(maxlen=delay_length)
#     delayed_actions = collections.deque(maxlen=delay_length)
#     delayed_hidden1 = collections.deque(maxlen=delay_length)
#     delayed_hidden2 = collections.deque(maxlen=delay_length)
#     delayed_x_values = collections.deque(maxlen=delay_length)
#
#     # Inner loop for each episode, check if we are within the last minute of close and if so, break
#     while True:
#         steps_done += 1
#         # Execute an action and get the next state, reward
#         action, hidden_state1, hidden_state2 = execute_action(state, hidden_state1, hidden_state2, steps_done,
#                                                               num_actions, Q_network)
#         next_state, x_value, done = env.perform_trade_step(action)
#
#         # Store the state, action and hidden states in the dequeue
#         delayed_states.append(state)
#         delayed_actions.append(action)
#         delayed_hidden1.append(hidden_state1)
#         delayed_hidden2.append(hidden_state2)
#         delayed_x_values.append(x_value)
#
#         if len(delayed_states) == delay_length:
#             # We now have enough steps to calculate a delayed reward
#             delayed_state = delayed_states[0]
#             delayed_action = delayed_actions[0]
#             delayed_h1 = delayed_hidden1[0]
#             delayed_h2 = delayed_hidden2[0]
#             delayed_x_value = delayed_x_values[0]
#
#             # Calculate the reward
#             reward = (x_value - delayed_x_value)
#             print(reward)
#
#             # Add the transition to the memory replay
#             memoryReplay.push(
#                 (delayed_state, delayed_h1, delayed_h2, delayed_action, state, reward, hidden_state1, hidden_state2))
#
#         # If the memory replay is full, sample a batch and update the Q-values
#         if len(memoryReplay) >= BATCH_SIZE:
#             transitions = memoryReplay.sample(BATCH_SIZE)
#             batch = Transition(*zip(*transitions))
#             update_Q_values(batch, Q_network, target_network, optimizer, architecture)
#
#         # If the number of steps is a multiple of C, update the target network
#         if steps_done % C == 0:
#             target_network.load_state_dict(Q_network.state_dict())
#
#         # If the number of steps is a multiple of Save_every, save the Q-network and the Target-network
#         if steps_done % Save_every == 0:
#             torch.save(Q_network.state_dict(), "models/{}.pth".format(env.ticker))
#             torch.save(target_network.state_dict(), "models/{}_target.pth".format(env.ticker))
#
#         # Update the current state
#         state = next_state

if __name__ == "__main__":
    ticker = "AAPL"
    live_env = LiveStockEnvironment(ticker=ticker, window_size=128, feature_size=32)
    live_env.load_state_deque()
    live_env.train()


