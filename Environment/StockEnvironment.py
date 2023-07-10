import csv
import math
import os
from collections import deque

from joblib import dump, load
import random

import numpy as np
import torch


class StockEnvironment:
    def __init__(self, starting_cash, starting_shares, window_size, feature_size, price_column, data=None,
                 scaled_data=None):
        # Define the action space as a list of integers from 0 to 10
        self.action_space = list(range(11))

        # Define the observation space as a tensor of zeros with shape (window_size, feature_size)
        self.observation_space = torch.zeros((window_size, feature_size), dtype=torch.float32).to(
            'cpu')  # Modify for your device, i.e., use 'cuda' for GPU

        # Set initial values for various variables
        self.episode_ended = False
        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.data = data
        self.scaled_data = scaled_data

        self.data_deque = deque(maxlen=window_size)
        self.scaled_data_deque = deque(maxlen=window_size)

        self.window_size = window_size
        self.feature_size = feature_size
        self.price_column = price_column
        self.current_step = 0
        self.current_price = 0
        self.current_cash = starting_cash
        self.current_shares = starting_shares
        self.current_portfolio_value = starting_cash

        if len (self.data) > 0:
            self.buy_and_hold_shares = (self.starting_cash / self.data[window_size,-3]) + self.starting_shares
            self.current_time_step_value = data[self.current_step]
        else:
            self.buy_and_hold_shares = 0
            self.current_time_step_value = 0
        self.batch_size = 524

        filename = './portfolio_values.csv'
        if os.path.isfile(filename):
            i = 1
            while os.path.isfile(f'portfolio_values_{i}.csv'):
                i += 1
            csv_file = open(f'portfolio_values_{i}.csv', 'w', newline='')
        else:
            csv_file = open(filename, 'w', newline='')

        writer = csv.writer(csv_file)
        writer.writerow(
            ['Step', 'Current Stock Price', 'Action', 'Buy and Hold Portfolio Value', 'DQN Agent Portfolio Value'])
        self.writer = writer

    def sample_action(self):
        """
        Returns a random action from the action space.
        """
        return random.choice(self.action_space)

    def get_current_price(self):
        """
        Returns the current price of the stock.
        """
        return self.data[self.current_step, self.price_column]

    def get_current_portfolio_value(self):
        """
        Returns the current portfolio value, which is the sum of the current cash and the current value of the shares.
        """
        return self.current_cash + self.current_shares * self.current_price

    def step(self, action):
        initial_portfolio_value = self.get_current_portfolio_value()

        if self.episode_ended:
            return self.reset()

        self.writer.writerow(
            [self.current_step, float(self.current_price), action, float(self.get_buy_and_hold_portfolio_value()),
             float(self.get_current_portfolio_value())])

        # Make the Trade by the action
        if action == 0:
            pass
        else:
            percentage_to_trade = ((action - 1) % 10 + 1) * 0.05

            if action < 6:  # Buy shares
                shares_to_buy = int((self.current_cash * percentage_to_trade) / self.current_price)
                if self.current_cash > self.current_price and shares_to_buy > 0:
                    # update cash and shares
                    self.current_cash -= shares_to_buy * self.current_price
                    self.current_shares += shares_to_buy
                else:
                    pass

            elif action < 11:  # Sell shares
                shares_to_sell = int(self.current_shares * percentage_to_trade)
                if self.current_shares > 0 and shares_to_sell > 0:
                    # update cash and shares
                    self.current_cash += shares_to_sell * self.current_price
                    self.current_shares -= shares_to_sell
                else:
                    pass

            else:
                raise ValueError("Action not recognized")

        self.current_step += 1

        self.data_deque.append(np.concatenate((self.data[self.current_step], [self.current_cash/
            self.current_portfolio_value, self.current_shares*self.current_price/self.current_portfolio_value])))
        self.scaled_data_deque.append(np.concatenate((self.scaled_data[self.current_step], [self
            .current_cash/self.current_portfolio_value, self.current_shares*self.current_price/self.current_portfolio_value])))

        done = False

        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            self.update_state()

        reward = (self.get_current_portfolio_value() - initial_portfolio_value)

        self.render(reward=reward, share_price=self.current_price)

        new_state = torch.tensor(np.array(self.scaled_data_deque), dtype=torch.float32).unsqueeze(0).to('cpu')

        done = torch.tensor(done, dtype=torch.bool).to('cpu')

        return new_state, reward, done

    def get_current_state(self):
        """
        Returns the current state of the environment.

        Returns:
            torch.tensor: The current state of the environment.
        """
        state = np.array(self.scaled_data_deque)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to('cpu')
        return state

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            torch.tensor: The current state of the environment.
        """
        self.current_step = self.window_size
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.current_price = self.get_current_price()
        self.buy_and_hold_shares = self.starting_shares + (self.starting_cash / self.get_current_price())

        return torch.tensor(self.get_current_state(), dtype=torch.float32).to('cpu')

    def soft_reset(self, new_data, new_scaled_data):
        self.current_step = 0
        self.data = new_data
        self.scaled_data = new_scaled_data
        return torch.tensor(self.get_current_state(), dtype=torch.float32).to('cpu')

    def render(self, reward, share_price):
        """
        Renders the current state of the environment.

        If the current step is within the length of the dataset, prints the current step, portfolio value, and buy and hold portfolio value.
        If the current step is outside the length of the dataset, prints "End of dataset".
        """
        if self.current_step <= len(self.data) - 1:
            print(
                f'Portfolio value: {self.get_current_portfolio_value():0.2f}, Buy and hold value: {self.get_buy_and_hold_portfolio_value():0.2f}, Reward: {reward:0.2f}, Share price: {share_price:0.2f}')
        else:
            print('End of dataset')

    # Update the current price, portfolio value, and portfolio value history
    def update_state(self):
        """
        Updates the current price, portfolio value, and portfolio value history.
        """
        self.current_price = self.get_current_price()
        self.current_portfolio_value = self.get_current_portfolio_value()

    # Returns the portfolio value if the agent buys and holds the stock from the beginning
    def get_buy_and_hold_portfolio_value(self):
        """
        Calculates the portfolio value if the agent buys and holds the stock from the beginning.

        Returns:
            float: The portfolio value.
        """
        # Calculate the portfolio value as the sum of starting cash and the value of starting shares
        # multiplied by the closing price of the stock at the current step
        return self.buy_and_hold_shares * self.data[self.current_step, 3]

    def initialize_state(self):
        """
        Push the first window_sizes to the data_deques
        """
        for i in range(self.window_size):
            self.data_deque.append(np.concatenate((self.data[i], [self.current_cash, self.current_shares])))
            self.scaled_data_deque.append(np.concatenate((self.scaled_data[i], [self.current_cash/self.current_portfolio_value, self.current_shares/self.current_portfolio_value])))

# ReplayMemory class for storing and sampling transitions
class ReplayMemory:
    def __init__(self, capacity):
        """
        Initializes the ReplayMemory object with a given capacity.

        Args:
            capacity (int): The maximum number of transitions that can be stored in the memory.
        """
        self.capacity = capacity
        self.memory = []  # list to store transitions
        self.position = 0  # current position in the memory

    def push(self, transition):
        """
        Saves a transition to the memory.

        Args:
            transition (tuple): A tuple containing the state, action, next state, reward, and done flag.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # add None to the list if memory is not full
        self.memory[self.position] = transition  # add the transition to the memory
        self.position = (self.position + 1) % self.capacity  # update the current position

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            A list of transitions of size batch_size.
        """
        return random.sample(self.memory, batch_size)

    def save_memory(self, symbol):
        """
        Saves the current memory to a file.
        """
        # Create the directory if it doesn't exist
        if not os.path.exists("ReplayMemoryCache"):
            os.makedirs("ReplayMemoryCache")

        # Save the replay memory to a file
        dump(self.memory, f"ReplayMemoryCache/{symbol}_replay_memory.joblib")

    def load_memory(self, symbol):
        """
        Loads the memory from a file.
        """
        # Load the replay memory from a file
        if os.path.exists(f"ReplayMemoryCache/{symbol}_replay_memory.joblib"):
            self.memory = load(f"ReplayMemoryCache/{symbol}_replay_memory.joblib")

    def __len__(self):
        """
        Returns the current size of the memory.

        Returns:
            The number of transitions currently stored in the memory.
        """
        return len(self.memory)


# EpsilonGreedyStrategy class for choosing actions
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        """
        Initializes the EpsilonGreedyStrategy object with a given start, end, and decay.

        Args:
            start (float): The starting exploration rate.
            end (float): The final exploration rate.
            decay (float): The rate at which the exploration rate decays over time.
        """
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        """
        Calculates the exploration rate for a given step.

        Args:
            current_step (int): The current step in the training process.

        Returns:
            The exploration rate for the current step.
        """
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

def epsilon_decay(steps_done, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    """
    Calculate decay of epsilon using the formula epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
    """
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)

