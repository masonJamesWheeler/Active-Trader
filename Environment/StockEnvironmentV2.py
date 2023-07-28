import csv
import math
import os
from collections import deque
import random

from Data.Data import get_and_process_data
from Data.DataLoader import DataLoader
from test import TiDE
import numpy as np
import heapq
import torch


class StockEnvironmentV2:
    def __init__(self, starting_cash, starting_shares, window_size, lookback_period, feature_size, price_column, data=None,
                 scaled_data=None):

        self.action_space = list(range(11))

        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.window_size = window_size
        self.lookback_period = lookback_period
        self.feature_size = feature_size
        self.price_column = price_column
        self.data = data
        self.scaled_data = scaled_data
        self.data_loader = DataLoader(self.data)

        self.current_step = self.lookback_period

        if len(self.data) > 0:
            self.current_price = self.data[self.current_step][self.price_column]
        else:
            self.current_price = 0

        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares

        self.TiDE = TiDE(input_size=30, hidden_size=50, num_encoder_layers=2, num_decoder_layers=2, output_dim=5, projected_dim=128)
        self.TiDE.load_weights(ticker='AAPL', live=False)

        self.finished = False

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
        '''
        Sample an action from the action space
        '''
        return random.choice(self.action_space)

    def get_current_price(self):
        '''
        Get the current price of the stock
        '''
        return self.data[self.current_step, self.price_column]

    def get_current_state(self):
        '''
        Get the current state of the environment
        '''
        data = self.data_loader.get_input_data(self.current_step)
        assert data[-1, 3] == self.scaled_data[self.current_step, 3], f'We seem to be mismatching data, prices are {data[-1, 3]} and {self.scaled_data[self.current_step, 3]}'

        prediction = self.model_prediction(data).detach().numpy()
        state = np.array(data)[:, 0:5]

        # [128, 5] + [128, 5] = [256, 5]
        state = np.concatenate((state, prediction), axis=0)
        # add a dimension for the batch size
        state = torch.tensor(np.expand_dims(state, axis=0))
        return state

    def get_current_portfolio_value(self):
        '''
        Get the current portfolio value
        '''
        return self.current_cash + self.current_shares * self.current_price

    def get_buy_and_hold_portfolio_value(self):
        """
        Calculates the portfolio value if the agent buys and holds the stock from the beginning.

        Returns:
            float: The portfolio value.
        """
        if self.current_step < len(self.data) - 1:
            return self.starting_cash + self.starting_shares * self.data[self.current_step, 3]
        else:
            return 0

    def reset(self):
        '''
        Reset the environment
        '''
        self.data_loader = DataLoader(self.scaled_data)
        self.current_step = self.lookback_period
        self.current_price = self.get_current_price()
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.finished = False
        return self.get_current_state()

    def soft_reset(self, data, scaled_data):
        '''
        Reset the environment
        '''
        self.data = data
        self.scaled_data = scaled_data
        self.data_loader = DataLoader(self.scaled_data)
        self.current_step = self.lookback_period
        self.finished = False
        return self.get_current_state()


    def model_prediction(self, state):
        '''
        Get the model prediction for the next 128 steps
        '''
        prediction = self.TiDE(state, state)
        return prediction

    def render(self, portfolio_value, buy_and_hold_portfolio_value, shares, share_price, reward, action, epsilon):
        '''
        Render the environment on the command line
        '''
        print(f'Portfolio Value: {portfolio_value.item()}, Buy and Hold Portfolio Value: {buy_and_hold_portfolio_value}, Shares:{shares},'
              f' Share Price: {share_price.item()}, Reward: {reward.item()}, Action: {action.item()}, Epsilon: {epsilon}')

    def step(self, action, epsilon):
        # Clip the action to the action space
        action = max(min(action, max(self.action_space)), min(self.action_space))

        # Observe the initial state
        initial_shares = self.current_shares
        initial_cash = self.current_cash
        initial_portfolio_value = self.get_current_portfolio_value()

        if self.finished:
            return self.reset()

        # Record the initial state
        self.writer.writerow(
            [self.current_step, float(self.current_price), action, float(self.get_buy_and_hold_portfolio_value()),
             float(initial_portfolio_value)])

        # Update the current price after the initial observation
        self.current_price = self.get_current_price()

        # Calculate the desired portfolio value based on the action
        if action < 6:
            desired_ratio = action * 0.2
        elif action < 11:
            desired_ratio = (action - 5) * (-0.05)
        else:
            raise ValueError("Action not recognized")

        # Logic for adjusting the portfolio cash and shares.
        # Maximum hold 100% of our value in shares and minimum -20% of our value in shares
        desired_shares = int(desired_ratio * initial_portfolio_value / self.current_price)

        # Buy or sell shares based on the desired shares
        if desired_shares >= self.current_shares:
            # Buy shares
            shares_to_buy = desired_shares - self.current_shares
            self.current_cash -= shares_to_buy * self.current_price
            self.current_shares += shares_to_buy
        else:
            # Sell shares
            shares_to_sell = self.current_shares - desired_shares
            self.current_cash += shares_to_sell * self.current_price
            self.current_shares -= shares_to_sell

        self.current_step += 1

        if self.current_step >= len(self.data) - 2:
            done = True
            self.finished = True  # Update the finished attribute
        else:
            done = False

        # The reward is calculated as the change in portfolio value after the trade
        reward = self.get_current_portfolio_value() - initial_portfolio_value
        next_state = self.get_current_state()

        self.render(self.get_current_portfolio_value(), self.get_buy_and_hold_portfolio_value(), self.current_shares,
                    self.current_price, reward, action, epsilon)

        return next_state, reward, done


if __name__ == '__main__':
    pass







