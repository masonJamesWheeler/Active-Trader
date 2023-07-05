import threading
import datetime
import math
import random
import time
from collections import deque
import csv
import pytz
import alpaca.data
import alpaca_trade_api
import numpy as np
import pandas as pd
import torch
from time import sleep
from alpaca.trading import OrderRequest, OrderSide, OrderType, TimeInForce, OrderStatus

from IB_API import IB_CLIENT
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from data import get_and_process_data
from get_fast_data import get_most_recent_data
from alpaca.trading.client import TradingClient
from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST, TimeFrame, APIError
import alpaca_trade_api as tradeapi

# ALPACA_KEY = "AKL0IN1Y4EG6A2Y37EQ1"
# ALPACA_SECRET_KEY = "NgyLanEH1hTo8r7xrlaBeSnefijyZLDpvvjxjAZl"
ALPHA_VANTAGE_API_KEY = "A5QND05S0W7CU55E"
PAPER_ALPACA_KEY = "PKWE20UZ10HFIC7QLMAX"
PAPER_ALPACA_SECRET_KEY = "SbPYFUJe4Ga9Nn96EF3DNIcKuatSlioXyRAbngOd"
base_url = 'https://paper-api.alpaca.markets'

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
        self.batch_size = 512
        self.ticker = ticker

        self.ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
        self.ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')

        self.data = deque(maxlen=window_size)
        self.portfolio_values = deque(maxlen=window_size)

        # self.update_indicators()
        base_url = 'https://paper-api.alpaca.markets'
        self.api = tradeapi.REST(PAPER_ALPACA_KEY, PAPER_ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.trading_client = TradingClient(PAPER_ALPACA_KEY, PAPER_ALPACA_SECRET_KEY)

        self.cash = 0
        self.account = 0
        self.positionSize = 0
        self.asset_price = 0
        self.portfolio_value = 0

        self.mode = "FAST"

    def run_listener(self):
        """
        Method to get the state vector which is fed to the trading agent. The state
        vector contains financial indicators for the current step along with ratios of
        cash and share portfolio. It also scales the features for better performance of
        the agent and constructs a new state vector.

        Returns:
            new_state (torch.tensor): Tensor representing the new state vector.
        """
        while True:
            print("run_listener: starting")

            snapshot = self.api.get_snapshot(self.ticker)
            self.account = account = self.api.get_account()
            self.portfolio_value = portfolioValue = float(account.portfolio_value)
            self.portfolio_values.append(portfolioValue)
            self.asset_price = assetPrice = float(snapshot.latest_trade.p)
            self.cash = cash  = float(account.buying_power)
            positions = self.api.list_positions()
            if len(positions) > 0:
                positionSize = float(positions[0].qty)
            else:
                positionSize = 0

            self.positionSize = positionSize

            newestDataPoint = np.array([
                snapshot.latest_trade.p,
                snapshot.latest_trade.s,
                snapshot.minute_bar.open,
                snapshot.minute_bar.high,
                snapshot.minute_bar.low,
                snapshot.minute_bar.close,
                snapshot.minute_bar.volume,
                snapshot.daily_bar.open,
                snapshot.daily_bar.high,
                snapshot.daily_bar.low,
                snapshot.daily_bar.close,
                snapshot.daily_bar.volume,
                snapshot.daily_bar.vwap,
                snapshot.prev_daily_bar.open,
                snapshot.prev_daily_bar.high,
                snapshot.prev_daily_bar.low,
                snapshot.prev_daily_bar.close,
                snapshot.prev_daily_bar.volume,
                snapshot.prev_daily_bar.vwap,
                portfolioValue,
                 cash,
                 positionSize], dtype=np.float32)
            self.data.append(newestDataPoint)
            if self.mode == "FAST":
                sleep(0.1)
                print(np.array(self.data).shape)
            else:
                sleep(1)

    def get_state(self):
        return torch.tensor(np.array(self.data).reshape((1, 128, 22)), dtype=torch.float32).to('cpu')

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
        # Cancel all orders
        self.api.cancel_all_orders()
        outOfBounds = False
        # Execute the trade based on the action
        if action == 0:
            pass
        else:
            for i in range(1, 6):
                percentage_to_trade = i * 0.05
                if action == i:  # Buy shares
                    if self.cash > self.asset_price:
                        shares_to_buy = int((self.cash * percentage_to_trade) / self.asset_price)
                        order_id = self.api.submit_order(symbol=self.ticker, qty=shares_to_buy, side='buy',
                                                         type='market', time_in_force='day')
                        print("buy order submitted")
                    else:
                        outOfBounds = True
            for i in range(6, 11):  # Sell shares
                percentage_to_trade = (i - 5) * 0.05
                if action == i:
                    current_shares = self.positionSize
                    if current_shares > 0:
                        shares_to_sell = int(self.positionSize * percentage_to_trade)
                        order_id = self.api.submit_order(symbol=self.ticker, qty=shares_to_sell, side='sell',
                                                         type='market', time_in_force='day')
                        print("sell order submitted")
                    else:
                        outOfBounds = True

        if action not in range(11):
            raise ValueError("Action not recognized")

        if action == 0 or outOfBounds:
            time.sleep(5)  # Wait for 10 seconds
            reward = self.get_reward()  # calculate reward
            next_state = self.get_state()  # calculate next state
        else:
            while True:
                if order_id.filled_at is not None or (datetime.datetime.now(pytz.UTC) - order_id.submitted_at).total_seconds() > 10:
                    # Wait until the order is filled or 10 seconds has passed
                    reward = self.get_reward()  # calculate reward
                    next_state = self.get_state()  # calculate next state
                    break

        return next_state, reward, False


    def get_reward(self):
        """
        Calculates the reward from the action taken in the current step.

        Returns:
            float: The reward for the current step.
        """
        # Calculate the reward as the mean of the most recent 1/2th of portfolio values
        if len(self.portfolio_values) < self.window_size / 2:
            return 0
        else:
            oldest_half = np.array(self.portfolio_values)[:int(self.window_size / 2)]
            newest_half = np.array(self.portfolio_values)[int(self.window_size / 2):]
            return torch.tensor((np.mean(newest_half) - np.mean(oldest_half))/ np.mean(oldest_half)*100).to('cpu')

    def render(self, reward):
        """
        Renders the environment.
        """
        print(f"Portfolio value: {self.portfolio_value}, Cash: {self.cash}, Position size: {self.positionSize}, Asset price: {self.asset_price}, Reward: {reward}")

    def start(self):
        """
        Starts the environment.
        """
        self.thread = threading.Thread(target=self.run_listener)
        self.thread.daemon = True
        self.thread.start()


if __name__ == '__main__':
    # Create a LiveTrader object
    trader = LiveStockEnvironment('AAPL', 128,30)
