import collections
import warnings
from collections import deque, namedtuple
from random import random, randrange

import numpy as np
import torch
import torch.optim as optim
from torch import device
device = device("cuda:0" if torch.cuda.is_available() else "cpu")
from Data.data import get_and_process_data, get_all_months
from Environment.StockEnvironment import StockEnvironment, ReplayMemory, EpsilonGreedyStrategy
from Environment.StockEnvironmentV2 import StockEnvironmentV2
from Models.DQN_Agent import DQN, update_Q_values

warnings.filterwarnings('ignore')

Transition = namedtuple('Transition',
                        ('state', 'hidden_state1', 'hidden_state2', 'action', 'next_state', 'reward',
                         'next_hidden_state1', 'next_hidden_state2'))


def initialize():
    """
    Initialize environment, DQN networks, optimizer and memory replay.
    """
    architecture = "RNN"
    starting_cash = 0
    starting_shares = 10000
    window_size = 128
    lookback_period = 400
    price_column = 3
    dense_size = 128
    dense_layers = 2
    feature_size = 35
    hidden_size = 128
    num_actions = 11
    dropout_rate = 0.2

    env = StockEnvironment(starting_cash=starting_cash, starting_shares=starting_shares, window_size=window_size,
                           feature_size=feature_size, price_column=price_column, data=[], scaled_data=[])

    memoryReplay = ReplayMemory(100000)

    Q_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                    architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                    dropout_rate=dropout_rate)

    target_network = DQN(input_size=feature_size, hidden_size=hidden_size, num_actions=num_actions,
                         architecture=architecture, dense_layers=dense_layers, dense_size=dense_size,
                         dropout_rate=dropout_rate)

    target_network.load_state_dict(Q_network.state_dict())
    optimizer = optim.Adam(Q_network.parameters(), lr=0.0001)

    hidden_state1, hidden_state2 = Q_network.init_hidden(1)

    return env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2

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


def main_loop(ticker, all_months, window_size=128, C=5, BATCH_SIZE=512, architecture='RNN'):
    """
    Run the main loop of DQN training.
    """
    # Set the interval, threshold, years, and months for data retrieval
    interval = '1min'

    # Initialize the environment, memory replay, Q-network, target network, optimizer, and hidden states
    env, memoryReplay, num_actions, Q_network, target_network, optimizer, last_hidden_state, current_hidden_state = initialize()

    Q_network.load_weights(False, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                           Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

    target_network.load_weights(True, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                                Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

    memoryReplay.load_memory(ticker)

    Q_network = Q_network.to(device)

    target_network = target_network.to(device)

    prev_hidden_state_1 = None
    prev_hidden_state_2 = None

    steps_done = 0
    iterator = 0
    reward = 0

    epsilon_strategy = EpsilonGreedyStrategy(start=1.00, end=0.01, decay=0.000001)

    # Loop through all tickers
    for month in all_months:
        iterator += 1
        # Retrieve and process data for the current ticker
        if iterator == 1:
            env.data, env.scaled_data, env.time_data, scaler = get_and_process_data(ticker, interval, window_size, month)
            env.initialize_state()
            state = env.reset()
            prev_hidden_state_1, prev_hidden_state_2 = Q_network.init_hidden(1)
        else:
            new_data, new_scaled_data, time_data, scaler = get_and_process_data(ticker, interval, window_size, month)
            print("RESET")
            state = env.soft_reset(new_data, new_scaled_data, time_data)

        done = False
        # Loop until the episode is done
        while not done:

            steps_done += 1

            epsilon = epsilon_strategy.get_exploration_rate(steps_done)

            # Execute an action and get the next state, reward, and done flag
            action, curr_hidden_state1, curr_hidden_state2, epsilon = execute_action(state, last_hidden_state, current_hidden_state, epsilon,
                                                                                       num_actions, Q_network)
            next_state, reward_delta, done = env.step(action, epsilon)

            reward += reward_delta

            if prev_hidden_state_1 is not None and prev_hidden_state_2 is not None:
                # Add the transition to the memory replay
                transition = Transition(state=state, hidden_state1=prev_hidden_state_1, hidden_state2=prev_hidden_state_2,
                                    action=action, next_state=next_state, reward=reward_delta, next_hidden_state1=curr_hidden_state1,
                                    next_hidden_state2=curr_hidden_state2)

                memoryReplay.push(transition)

            # If the memory replay is full, sample a batch and update the Q-values every 4 steps
            if len(memoryReplay) >= BATCH_SIZE and steps_done % 4 == 0:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer, architecture)

            # If the number of steps is a multiple of C (100), update the target network
            if steps_done % 100 == 0:
                target_network.load_state_dict(Q_network.state_dict())

            state = next_state

        Q_network.save_weights(False, ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                               Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

        target_network.save_weights(True, ticker, Q_network.dense_layers_num, Q_network.dense_size,
                                    Q_network.hidden_size,
                                    Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

        memoryReplay.save_memory(ticker)



if __name__ == "__main__":
    ticker = "AAPL"
    start_year = 2000
    start_month = 4
    end_year = 2023
    end_month = 7

    all_months = get_all_months(start_year, start_month, end_year, end_month)
    main_loop(ticker="AAPL", all_months=all_months)
