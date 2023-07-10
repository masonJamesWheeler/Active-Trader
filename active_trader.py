import pickle
import warnings
from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim

from Data.Data import get_and_process_data, get_all_months
from Environment.StockEnvironment import StockEnvironment, ReplayMemory, epsilon_decay
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
    starting_cash = 10000
    starting_shares = 1000
    window_size = 128
    price_column = 3
    dense_size = 256
    dense_layers = 2
    feature_size = 32
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
    optimizer = optim.Adam(Q_network.parameters())

    hidden_state1, hidden_state2 = Q_network.init_hidden(1)

    return env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2


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


def main_loop(ticker, all_months, window_size=128, C=10, BATCH_SIZE=512, architecture='RNN'):
    """
    Run the main loop of DQN training.
    """
    # Set the interval, threshold, years, and months for data retrieval
    interval = '1min'

    # Initialize the environment, memory replay, Q-network, target network, optimizer, and hidden states
    env, memoryReplay, num_actions, Q_network, target_network, optimizer, hidden_state1, hidden_state2 = initialize()
    steps_done = 0
    iterator = 0

    # Loop through all tickers
    for month in all_months:
        iterator += 1
        # Retrieve and process data for the current ticker
        if iterator == 1:
            env.data, env.scaled_data, scaler = get_and_process_data(ticker, interval, window_size, month)
            env.initialize_state()
            state = env.reset()
            hidden_state1, hidden_state2 = Q_network.init_hidden(1)
        else:
            new_data, new_scaled_data, scaler = get_and_process_data(ticker, interval, window_size, month)
            print("RESET")
            state = env.soft_reset(new_data, new_scaled_data)
            env.current_step = 0

        done = False
        # Loop until the episode is done
        while not done:
            steps_done += 1
            # Execute an action and get the next state, reward, and done flag
            action, hidden_state1, hidden_state2 = execute_action(state, hidden_state1, hidden_state2, steps_done,
                                                                  num_actions, Q_network)
            next_state, reward, done = env.step(action)
            # Render the environment
            # Add the transition to the memory replay
            memoryReplay.push(
                (state, hidden_state1, hidden_state2, action, next_state, reward, hidden_state1, hidden_state2))

            # If the memory replay is full, sample a batch and update the Q-values
            if len(memoryReplay) >= BATCH_SIZE:
                transitions = memoryReplay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, Q_network, target_network, optimizer, architecture)

            # If the number of steps is a multiple of C, update the target network
            if steps_done % C == 0:
                target_network.load_state_dict(Q_network.state_dict())

            state = next_state

        Q_network.save_weights(False, ticker)
        target_network.save_weights(True, ticker)
        memoryReplay.save_memory(ticker)


if __name__ == "__main__":
    ticker = "AAPL"
    start_year = 2018
    start_month = 1
    end_year = 2023
    end_month = 7

    all_months = get_all_months(start_year, start_month, end_year, end_month)

    main_loop(ticker="AAPL", all_months=all_months)
