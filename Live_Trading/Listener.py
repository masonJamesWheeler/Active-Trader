import collections
import os
import warnings
from collections import deque, namedtuple
from random import random, randrange
import alpaca_trade_api as tradeapi
import numpy as np
import torch
import torch.optim as optim
from torch import device
device = device("cuda:0" if torch.cuda.is_available() else "cpu")
from Data.data import get_and_process_data, get_all_months
from Data.Get_Fast_Data import get_most_recent_data2
from Environment.StockEnvironment import ReplayMemory, epsilon_decay
from Models.DQN_Agent import DQN, update_Q_values
from PostGreSQL.Database import *
from Training.Utils import *
from datetime import datetime
import time

# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets',
                         api_version='v2')

warnings.filterwarnings('ignore')

Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward',
                         'next_hidden_state1', 'next_hidden_state2'))

ticker = "AAPL"
month = "2023-07"
table_name = f'ticker_{ticker}_data'

# Load the replay memory
replay_memory = ReplayMemory(50000)
replay_memory.load_memory(ticker)

# Load the DQNs
architecture = "RNN"
feature_size = 35
num_actions = 11
dropout_rate = 0.2
hidden_size = 128
dense_size = 128
batch_size = 512

memoryReplay = ReplayMemory(50000)

Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                architecture=architecture, dense_layers=dense_size, dense_size=dense_size,
                dropout_rate=dropout_rate)

target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                     architecture=architecture, dense_layers=dense_size, dense_size=dense_size,
                     dropout_rate=dropout_rate)

target_network.load_state_dict(Q_network.state_dict())

Q_network.load_weights(False, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                       Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

target_network.load_weights(True, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                            Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

optimizer = optim.Adam(Q_network.parameters())

last_hidden_state1, last_hidden_state2 = Q_network.init_hidden(1)

# Move the model to the device
Q_network.to(device)
target_network.to(device)

# Definitions of the price, share, and cash variables
last_state = get_latest_row(table_name)
last_timestamp = last_state[0]
last_price = last_state[4]
last_shares = last_state[-1]

action = 0
steps_done = 0

def test1():
    ohlvc = get_latest_ohlcv(table_name)
    timestamp = ohlvc[0]
    open = ohlvc[1]
    high = ohlvc[2]
    low = ohlvc[3]
    close = ohlvc[4]
    volume = ohlvc[5]
    print(f'Latest OHLCV: {timestamp}, {open}, {high}, {low}, {close}, {volume}')


def update_portfolio_values():
    # Get the account information
    account = api.get_account()

    # Check if the account data is valid
    if account is None or not hasattr(account, 'portfolio_value'):
        print("Invalid account data. Exiting...")
        return False

    # Update the portfolio and account data stored in the environment
    portfolio_value = float(account.portfolio_value)
    share_value = float(account.position_value)
    cash = float(account.cash)

    return portfolio_value, share_value, cash


def get_new_state():
    new_state = np.array(get_latest_n_rows(table_name, 127))
    print(f'New state: {new_state.shape}')
#   remove the [0], [-1], and [-2] columns
    new_state = np.delete(new_state, [0, -1, -2], axis=1)

#   add the most recent data to the new state
    new_state = np.append(new_state, get_most_recent_data2(symbol=ticker, interval='1min', window_size=128, month=month))



def listener(steps_done=0, C=10, epsilon=0.01):
    current_minute = datetime.now().minute

    while True:

        if datetime.now().minute != current_minute:
            current_minute = datetime.now().minute

            steps_done += 1

            action, hidden_state1, hidden_state2, epsilon = execute_action(state=last_state, hidden_state1=last_hidden_state1, hidden_state2=last_hidden_state2,
                                                                           epsilon= epsilon, num_actions=11, Q_network=Q_network)


            # Gather the latest data
            new_state = get_new_state()

            new_price = new_state[3]
            reward = last_shares * (new_price - last_price)

            transition = Transition(state=last_state, hidden_state1=last_hidden_state1, hidden_state2=last_hidden_state2,
                                    action=action, next_state=new_state, reward=reward,
                                    next_hidden_state1=hidden_state1, next_hidden_state2=hidden_state2)

            # Update the replay memory
            replay_memory.push(transition)

            if len(memoryReplay) >= batch_size:
                # Update the Q-network
                transitions = memoryReplay.sample(batch_size)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer, architecture)

            # If the number of steps is a multiple of C, update the target network
            if steps_done % C == 0:
                target_network.load_state_dict(Q_network.state_dict())

            last_state = new_state
            last_price = new_price
            last_shares = new_shares

            last_action = action



        time.sleep(1)

if __name__ == '__main__':
    listener()
