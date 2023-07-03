import datetime
import math
import random
import threading
import time
from collections import deque
from time import sleep
import csv
import numpy as np
import pandas as pd
import torch
from IB_API import IB_CLIENT
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from data import get_and_process_data
from get_fast_data import get_most_recent_data

API_KEY = "A5QND05S0W7CU55E"



class LiveStockEnvironment:
    def __init__(self, ticker, window_size, feature_size, price_column):
        # Define the action space as a list of integers from 0 to 10
        self.action_space = list(range(11))

        # Define the observation space as a tensor of zeros with shape (window_size, feature_size+2)
        self.observation_space = torch.zeros((window_size, feature_size + 2), dtype=torch.float32).to(
            'cpu')

        # Set initial values for various variables
        self.episode_ended = False
        self.window_size = window_size
        self.feature_size = feature_size
        self.price_column = price_column
        self.current_step = 0
        self.current_price = 0
        self.batch_size = 524
        self.ticker = ticker
        self.data = deque(maxlen=window_size)

        self.ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
        self.ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')
        self.client = IB_CLIENT()

        _, __, self.scaler = get_and_process_data(ticker=ticker, interval='1min', window_size=window_size, threshold=0.01,
                                                  years=2,months=12)
        self.cash = self.client.get_cash_balance()
        self.shares = self.client.get_shares(ticker)
        self.portfolio_value = self.client.get_portfolio_value()

    def sample_action(self):
        """
        Returns a random action from the action space.
        """
        return random.choice(self.action_space)

    def get_current_price(self):
        """
        Returns the current price of the stock.
        """
        return self.ts.get_quote_endpoint(symbol='AAPL')[0]['05. price']

    def get_current_portfolio_value(self):
        """
        Returns the current portfolio value, which is the sum of the current cash and the current value of the shares.
        """
        return self.client.get_portfolio_value()

    def step(self, action, share_price, data:deque):
        # Record the current minute
        current_minute = datetime.datetime.now().minute
        initial_portfolio_value = self.client.get_portfolio_value()
        self.cash = self.client.get_cash_balance()
        # Make the Trade by the action
        if action == 0:
            pass
        elif action == 1:  # Use 5% of cash value to buy a share
            if self.cash > share_price:
                shares_to_buy = int((self.cash * 0.05) / share_price)
                self.client.buy_shares_mkt(self.ticker, shares_to_buy)
            else:
                pass
        elif action == 2:  # Use 10% of cash value to buy a share
            if self.cash > share_price:
                shares_to_buy = int((self.cash * 0.10) / share_price)
                self.client.buy_shares_mkt(self.ticker, shares_to_buy)
            else:
                pass
        elif action == 3:  # Use 15% of cash value to buy a share
            if self.cash > share_price:
                shares_to_buy = int((self.cash * 0.15) / share_price)
                self.client.buy_shares_mkt(self.ticker, shares_to_buy)
            else:
                pass
        elif action == 4:  # Use 20% of cash value to buy a share
            if self.cash > share_price:
                shares_to_buy = int((self.cash * 0.20) / share_price)
                self.client.buy_shares_mkt(self.ticker, shares_to_buy)
            else:
                pass
        elif action == 5:  # Use 25% of cash value to buy a share
            if self.cash > share_price:
                shares_to_buy = int((self.cash * 0.25) / share_price)
                self.client.buy_shares_mkt(self.ticker, shares_to_buy)
            else:
                pass
        elif action == 6:  # Sell 5% of shares
            current_shares = self.client.get_shares(self.ticker)
            if current_shares > 0:
                shares_to_sell = int(current_shares * 0.05)
                self.client.sell_shares_mkt(self.ticker, shares_to_sell)
            else:
                pass
        elif action == 7:  # Sell 10% of shares
            current_shares = self.client.get_shares(self.ticker)
            if current_shares > 0:
                shares_to_sell = int(current_shares * 0.10)
                self.client.sell_shares_mkt(self.ticker, shares_to_sell)
            else:
                pass
        elif action == 8:  # Sell 15% of shares
            current_shares = self.client.get_shares(self.ticker)
            if current_shares > 0:
                shares_to_sell = int(current_shares * 0.15)
                self.client.sell_shares_mkt(self.ticker, shares_to_sell)
            else:
                pass
        elif action == 9:  # Sell 20% of shares
            current_shares = self.client.get_shares(self.ticker)
            if current_shares > 0:
                shares_to_sell = int(current_shares * 0.20)
                self.client.sell_shares_mkt(self.ticker, shares_to_sell)
            else:
                pass
        elif action == 10:  # Sell 25% of shares
            current_shares = self.client.get_shares(self.ticker)
            if current_shares > 0:
                shares_to_sell = int(current_shares * 0.25)
                self.client.sell_shares_mkt(self.ticker, shares_to_sell)
            else:
                pass
        else:
            raise ValueError("Action not recognized")

        # Waste time until the time doesn't equal the current minute
        while datetime.datetime.now().minute == current_minute:
            pass

        reward = self.client.get_portfolio_value() - initial_portfolio_value
        self.data.append(get_most_recent_data(self.ticker, '1min', API_KEY, 128))
    #   convert the deque to a numpy array
        new_state = torch.tensor(self.scaler.transform(np.array(self.data)), dtype=torch.float32)
        done = False
        return new_state, reward, done

    def render(self, reward):
        """
        Renders the current state of the environment.

        If the current step is within the length of the dataset, prints the current step, portfolio value, and buy and hold portfolio value.
        If the current step is outside the length of the dataset, prints "End of dataset".
        """
        print(
            f'Portfolio Value: {self.client.get_portfolio_value()}')

