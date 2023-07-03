import math
import random
from collections import deque
from time import sleep
import csv
import numpy as np
import pandas as pd
import torch
import os



class StockEnvironment:
    def __init__(self, starting_cash, starting_shares, data, scaled_data, window_size, feature_size, price_column,
                 reward_function):
        # Define the action space as a list of integers from 0 to 10
        self.action_space = list(range(11))

        # Define the observation space as a tensor of zeros with shape (window_size, feature_size+2)
        self.observation_space = torch.zeros((window_size, feature_size + 2), dtype=torch.float32).to(
            'cpu')  # Modify for your device, i.e., use 'cuda' for GPU

        # Set initial values for various variables
        self.episode_ended = False
        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.data = data
        self.scaled_data = scaled_data
        self.window_size = window_size
        self.feature_size = feature_size
        self.price_column = price_column
        self.current_step = 0
        self.current_price = 0
        self.current_cash = starting_cash
        self.current_shares = starting_shares
        self.current_portfolio_value = starting_cash
        self.buy_and_hold_shares = (self.starting_cash/ self.data[0, -1, self.price_column]) + self.starting_shares
        self.batch_size = 524
        self.current_time_step_value = data[self.current_step]
        self.reward_function = reward_function


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
        return self.data[self.current_step, -1, self.price_column]

    def get_current_portfolio_value(self):
        """
        Returns the current portfolio value, which is the sum of the current cash and the current value of the shares.
        """
        return self.current_cash + self.current_shares * self.current_price

    def step(self, action):
        initial_portfolio_value = self.get_current_portfolio_value()

    #     Check if the episode is over
        if self.episode_ended:
            return self.reset()
        self.writer.writerow([self.current_step, float(self.current_price) , action, float(self.get_buy_and_hold_portfolio_value()), float(self.get_current_portfolio_value())])
    # Make the Trade by the action
        if action == 0:
            pass
        elif action == 1: # Use 5% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.05)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 2: # Use 10% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.10)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 3: # Use 15% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.15)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 4: # Use 20% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.20)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 5: # Use 25% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.25)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 6: # Sell 5% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.05))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 7: # Sell 10% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.10))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 8: # Sell 15% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.15))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 9: # Sell 20% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.20))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 10: # Sell 25% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.25))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        else:
            raise ValueError("Action not recognized")

        # add the current price to the price history
        self.price_history.append(self.current_price)
        self.current_step += 1

        done = False
        if self.current_step >= len(self.data) -1:
            done = True
        else:
            self.update_state()

        reward = self.get_current_portfolio_value() - initial_portfolio_value
        # reward = (((self.get_current_portfolio_value() - initial_portfolio_value) / initial_portfolio_value)) * 1000 # Scale reward to be between -100 and 100
        self.render(reward=reward, share_price = self.current_price)
        new_state = torch.tensor(self.get_current_state(), dtype=torch.float32).to('cpu')
        # done to tensor
        done = torch.tensor(done, dtype=torch.bool).to('cpu')

        return new_state, reward, done
    
    def get_current_state(self):
        """
        Returns the current state of the environment.

        Returns:
            torch.tensor: The current state of the environment.
        """
        state = np.array(self.scaled_data[self.current_step], dtype=np.float32)
        # add the cash/portfolio value ratio and the share_value/portfolio value ratio at the end
        cash_portfolio_ratio = self.current_cash / self.get_current_portfolio_value()
        share_value_ratio = self.current_price * self.current_shares / self.get_current_portfolio_value()

        state_with_ratios = np.zeros(shape=(1, self.window_size, state.shape[1]+2), dtype=np.float32)
        state_with_ratios[0, :, :-2] = state
        state_with_ratios[0, :, -2] = cash_portfolio_ratio
        state_with_ratios[0, :, -1] = share_value_ratio

        state = torch.tensor(state_with_ratios, dtype=torch.float32).to('cpu')
        return state


    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            torch.tensor: The current state of the environment.
        """
        self._episode_ended = False
        self.current_step = 0
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.current_portfolio_value_history = []
        self.current_portfolio_value_history.append(self.starting_cash)
        self.current_price = self.get_current_price()

        # Fill all the histories with zeroes
        self.price_history = [0 for i in range(self.window_size)]
        self.portfolio_value_history = [0 for i in range(self.window_size)]
        self.cash_history = [0 for i in range(self.window_size)]
        self.action_history = [0 for i in range(self.window_size)]
        self.shares_history = [0 for i in range(self.window_size)]

        return torch.tensor(self.get_current_state(), dtype=torch.float32).to('cpu')

    def render(self, reward, share_price):
        """
        Renders the current state of the environment.

        If the current step is within the length of the dataset, prints the current step, portfolio value, and buy and hold portfolio value.
        If the current step is outside the length of the dataset, prints "End of dataset".
        """
        if self.current_step <= len(self.data) -1:
            print(f'Step: {self.current_step}, Portfolio Value: {self.current_portfolio_value}, vs. Buy and Hold: {self.get_buy_and_hold_portfolio_value()}, Reward: {reward}', f'Share Price: {share_price}')
        else:
            print('End of dataset')

    # Update the current price, portfolio value, and portfolio value history
    def update_state(self):
        """
        Updates the current price, portfolio value, and portfolio value history.
        """
        self.current_price = self.get_current_price()
        self.current_portfolio_value = self.get_current_portfolio_value()
        self.current_portfolio_value_history.append(self.current_portfolio_value)

    # Returns the portfolio value if the agent buys and holds the stock from the beginning
    def get_buy_and_hold_portfolio_value(self):
        """
        Calculates the portfolio value if the agent buys and holds the stock from the beginning.

        Returns:
            float: The portfolio value.
        """
        # Calculate the portfolio value as the sum of starting cash and the value of starting shares
        # multiplied by the closing price of the stock at the current step
        return self.buy_and_hold_shares * self.data[self.current_step, -1, 3]

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
