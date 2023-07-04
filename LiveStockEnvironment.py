import asyncio
import datetime
import math
import random
import threading
import time
from collections import deque
import csv
import numpy as np
import pandas as pd
import torch
from time import sleep
from IB_API import IB_CLIENT
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from data import get_and_process_data
from get_fast_data import get_most_recent_data

API_KEY = "A5QND05S0W7CU55E"

class LiveStockEnvironment:
    def __init__(self, ticker, window_size, feature_size):
        # Define the action space as a list of integers from 0 to 10
        self.action_space = list(range(11))

        # Define the observation space as a tensor of zeros with shape (window_size, feature_size+2)
        self.observation_space = torch.zeros((window_size, feature_size + 2), dtype=torch.float32).to(
            'cpu')

        # Set initial values for various variables
        self.episode_ended = False
        self.window_size = window_size
        self.feature_size = feature_size
        self.current_step = 0
        self.current_price = 0
        self.batch_size = 512
        self.ticker = ticker

        self.ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
        self.ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')
        self.client = IB_CLIENT()

        self.past_data, self.past_scaled_data, self.scaler = get_and_process_data(ticker=ticker, interval='1min', window_size=window_size, threshold=0.01,
                                                  years=2,months=12, api_key=API_KEY)
        self.data = deque(maxlen=window_size)
        self.cash = self.client.get_cash_balance()
        self.shares = self.client.get_shares(ticker)
        self.portfolio_value = self.client.get_portfolio_value()
        self.indicators = np.zeroes([window_size])

        self.update_indicators()

    def get_new_state(self):
        """
        Method to get the state vector which is fed to the trading agent. The state
        vector contains financial indicators for the current step along with ratios of
        cash and share portfolio. It also scales the features for better performance of
        the agent and constructs a new state vector.

        Returns:
            new_state (torch.tensor): Tensor representing the new state vector.
        """

        # Compute current portfolio value
        portfolio_value = self.client.get_portfolio_value()

        # Compute the ratio of cash balance to portfolio value
        cash_portfolio_ratio = self.client.get_cash_balance() / portfolio_value

        # Compute the ratio of share portfolio value to portfolio value
        share_portfolio_ratio = (portfolio_value - cash_portfolio_ratio) / portfolio_value

        # Concatenate these ratios to the indicators
        current_indicators = self.indicators
        current_indicators = np.append(current_indicators, cash_portfolio_ratio)
        current_indicators = np.append(current_indicators, share_portfolio_ratio)

        # Append current_indicators to data
        self.data.append(current_indicators)

        # Create a numpy array of data
        current_state = np.array(self.data)

        # Split the state into two parts
        first_30_features = current_state[:, :30]  # All rows, first 30 columns
        last_2_features = current_state[:, 30:]  # All rows, last 2 columns

        # Save original shape for reshaping after scaling
        original_shape = first_30_features.shape

        # Flatten first_30_features for scaling
        first_30_features_flattened = first_30_features.flatten()

        # Scale the features using the saved scaler
        scaled_features = self.scaler.transform(first_30_features_flattened.reshape(1, -1))

        # Reshape the scaled features back to their original shape
        reshaped_scaled_features = scaled_features.reshape(original_shape)

        # Concatenate the scaled features and last 2 features to get the final feature set
        final_features = np.concatenate((reshaped_scaled_features, last_2_features), axis=1)

        # Convert the final_features to PyTorch tensor and move it to CPU
        new_state = torch.tensor(final_features, dtype=torch.float32).to('cpu')

        return new_state

    def update_indicators(self):
        """
        Updates the indicators every 60 seconds
        """
        while True:
            self.indicators = np.array(get_most_recent_data(self.ticker, '1min', API_KEY, 128)).reshape(1, -1)
            sleep(60)

    def sample_action(self):
        """
        Returns a random action from the action space.
        """
        return random.choice(self.action_space)

    def get_current_price(self):
        """
        Returns the current price of the stock.
        """
        return float(self.ts.get_quote_endpoint(symbol='AAPL')[0]['05. price'])

    def get_current_portfolio_value(self):
        """
        Returns the current portfolio value, which is the sum of the current cash and the current value of the shares.
        """
        return self.client.get_portfolio_value()

    def perform_trade_step(self, action):
        """
        Executes a trading step based on the provided action.

        The actions are as follows:
            0: Hold
            1-5: Buy shares using 5-25% of cash
            6-10: Sell 5-25% of shares

        The reward is calculated as the difference between the new portfolio value and the initial portfolio value.
        A new state is returned, which represents the current indicators.

        Args:
            action (int): The action to take.

        Returns:
            tuple: The new state, the reward from the action, and a boolean indicating if trading is done.
        """
        current_share_price = self.get_current_price()
        initial_portfolio_value = self.get_current_portfolio_value()
        new_portfolio_value = initial_portfolio_value
        self.cash = self.client.get_cash_balance()

        # Execute the trade based on the action
        if action == 0:
            trade_executed = True
        else:
            for i in range(1, 11):
                percentage_to_trade = i * 0.05
                if action == i:  # Buy shares
                    if self.cash > current_share_price:
                        shares_to_buy = int((self.cash * percentage_to_trade) / current_share_price)
                        trade_id = self.client.buy_shares_mkt(self.ticker, shares_to_buy)
                elif action == i + 5:  # Sell shares
                    current_shares = self.client.get_shares(self.ticker)
                    if current_shares > 0:
                        shares_to_sell = int(current_shares * percentage_to_trade)
                        trade_id = self.client.sell_shares_mkt(self.ticker, shares_to_sell)

        if action not in range(11):
            raise ValueError("Action not recognized")

        # Start a timer
        start_time = time.time()

        portfolio_updated = False
        trade_executed = False
        while not trade_executed:
            if self.client.executed_last_trade(trade_id):
                trade_executed = True

        # Check if portfolio value is updated and at least 2 seconds have passed
        while not portfolio_updated and time.time() - start_time > 2:
            portfolio_updated, new_portfolio_value = self.client.porfolio_value_updated(initial_portfolio_value)

        # Calculate the reward
        reward = new_portfolio_value - initial_portfolio_value
        new_state = self.get_new_state()

        trade_complete = False
        self.render(reward, initial_portfolio_value, new_portfolio_value, action)
        return new_state, reward, trade_complete

    def render(self, reward, past_portfolio_value, portfolio_value, past_action):
        """
        Renders the current state of the environment.

        Prints the old portfolio value, the action taken, and the new portfolio value, and the reward
        """
        print(
            f'Old Portfolio Value: {past_portfolio_value} | Action Taken: {past_action} | New Portfolio Value: {portfolio_value} | Reward: {reward}')

    def initalize(self):
        """
        Initializes the live trader by getting the current cash balance and portfolio value. It calculates
        the cash to portfolio ratio and share to portfolio ratio and appends these to the past data. The
        method then waits until the next minute before starting live trading.

        Note:
            This function assumes that self.client is an object with methods to fetch current cash balance
            and portfolio value. It also assumes that self.past_data is a numpy array representing past
            trading data.
        """

        # Get current date and time
        start_time = datetime.datetime.now()

        # Fetch current cash balance
        cash_value = self.client.get_cash_balance()

        # Fetch current portfolio value
        portfolio_value = self.client.get_portfolio_value()

        # Compute value of shares in the portfolio
        share_value = portfolio_value - cash_value

        # Compute ratio of cash value to portfolio value
        cash_portfolio_ratio = cash_value / portfolio_value

        # Compute ratio of share value to portfolio value
        share_portfolio_ratio = share_value / portfolio_value

        # Get the last window of past data
        last_window = self.past_data[-1]

        # Iterate over each element in the last window of past data
        for i in range(last_window.shape[0]):
            # Get the current array
            array = self.past_data[-1, i]

            # Append the cash portfolio ratio to the current array
            array = np.append(array, cash_portfolio_ratio)

            # Append the share portfolio ratio to the current array
            array = np.append(array, share_portfolio_ratio)

            # Append the updated array to the deque
            self.data.append(array)

        # Print initialization completion message
        print('Live Trader Initialized')

        # Pause execution until the start of the next minute or at least 60 seconds have passed since start_time
        while datetime.datetime.now().minute == start_time.minute and (
                datetime.datetime.now() - start_time).seconds < 60:
            pass






