import logging
import threading
import datetime
import time
from collections import deque
import csv
import pytz
import numpy as np
import pandas as pd
import torch
from time import sleep

from alpaca.trading import TradingClient
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
from Data.Get_Fast_Data import get_most_recent_data

# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

class LiveStockEnvironment:
    def __init__(self, ticker, window_size, feature_size):
        # Define the action space as a list of integers from 0 to 10
        self.action_space = list(range(11))

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
        self.data_queue = deque(maxlen=window_size)
        self.alpha_vantage_data = None

        # self.update_indicators()
        self.api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets', api_version='v2')
        self.trading_client = TradingClient(paper_alpaca_key, paper_alpaca_secret_key)

        self.cash = 0
        self.account = 0
        self.positionSize = 0
        self.asset_price = 0
        self.portfolio_value = 0

        self.thread1 = threading.Thread(target=self.run_listener)
        self.thread1.daemon = True

        self.thread2 = threading.Thread(target=self.alpha_vantage_update)
        self.thread2.daemon = True
        self.mode = "FAST"

    def run_listener(self):
        """
        Method to get the state vector which is fed to the trading agent. The state
        vector contains financial indicators for the current step along with ratios of
        cash and share portfolio.
        """
        while True:
            time.sleep(0.1 if self.mode == "FAST" else 1.5)
            try:
                if not self.data_queue:
                    continue

                alpha_vantage_data = self.data_queue[-1]
                if alpha_vantage_data is None:
                    continue

                snapshot = self.api.get_snapshot(self.ticker)
                if snapshot is None or snapshot.latest_trade is None or snapshot.latest_trade.p is None:
                    continue

                self.account = account = self.api.get_account()
                if account is None or account.portfolio_value is None or account.buying_power is None:
                    continue

                self.portfolio_value = portfolio_value = float(account.portfolio_value)
                self.portfolio_values.append(portfolio_value)
                self.asset_price = asset_price = float(snapshot.latest_trade.p)
                self.cash = cash = float(account.buying_power)

                positions = self.api.list_positions()
                if not positions or positions[0]._raw['qty'] is None:
                    continue

                self.positionSize = position_size = int(positions[0]._raw['qty'])

                newest_data_point = self.create_new_data_point(snapshot, cash, portfolio_value, asset_price,
                                                               position_size, alpha_vantage_data)
                self.data.append(newest_data_point)
            except Exception as e:
                logging.error(f"Run listener failed: {e}")

    @staticmethod
    def create_new_data_point(snapshot, cash, portfolio_value, asset_price, position_size, alpha_vantage_data):
        """
        Helper method to create the new data point.
        """
        data_point = np.array([
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
            cash / portfolio_value,
            position_size * asset_price / portfolio_value], dtype=np.float32)

        newest_data_point = np.concatenate((data_point, alpha_vantage_data), axis=0)

        return newest_data_point

    def alpha_vantage_update(self):
            '''
            Function that retrieves the most recent data from Alpha Vantage and updates the current data.
            '''
            while True:
                try:
                    self.alpha_vantage_data = get_most_recent_data("AAPL", "1min")
                    self.data_queue.append(self.alpha_vantage_data)  # Put the new data into the queue
                    time.sleep(60)
                except Exception as e:
                    logging.error(f"Alpha Vantage update failed: {e}")
                    time.sleep(300)  # Wait for a longer period before trying again if there's a failure

    def get_state(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(0).to(device)

    def perform_trade_step(self, action):
        """
        Executes a trading step based on the provided action.
        Args:
            action (int): The action to take.
        Returns:
            tuple: The new state, the reward from the action, and a boolean indicating if trading is done.
        """
        self._cancel_all_orders()

        if action not in range(11):
            raise ValueError("Action not recognized")

        if action == 0:
            outOfBounds = True
            order = None
            sleep(2)
        else:
            outOfBounds, order = self._execute_trade(action)

        reward, next_state = self._compute_reward_and_next_state(order, outOfBounds)

        self.render(reward)
        return next_state, reward, False

    def _cancel_all_orders(self):
        """
        Cancel all open orders.
        """
        self.api.cancel_all_orders()

    def _execute_trade(self, action):
        """
        Execute the trade based on the action.
        Args:
            action (int): The action to take.
        Returns:
            bool: True if trading is not possible, False otherwise.
        """
        order = None
        for i in range(1, 11):
            percentage_to_trade = ((i - 1) % 5 + 1) * 0.05

            if action == i:
                if i < 6:
                    side = 'buy'
                    quantity = int((self.cash * percentage_to_trade) / self.asset_price)
                    if self.cash <= self.asset_price or quantity <= 0:
                        return True, None
                else:
                    side = 'sell'
                    quantity = int(self.positionSize * percentage_to_trade)
                    if self.positionSize <= 0 or quantity <= 0:
                        return True, None

                try:
                    order = self.api.submit_order(symbol=self.ticker, qty=quantity, side=side,
                                                  type='market', time_in_force='day')
                    print(f"{side} order submitted")
                    break
                except Exception as e:
                    logging.error(f"Order submission failed: {e}")
                    return True, None

        return False, order

    def _compute_reward_and_next_state(self, order, outOfBounds):
        """
        Compute the reward and the next state.
        Args:
            outOfBounds (bool): Whether trading was possible.
        Returns:
            tuple: The reward and the next state.
        """
        if outOfBounds:
            reward = self.get_reward()  # calculate reward
            next_state = self.get_state()  # calculate next state
        else:
            while True:
                if order.filled_at is not None or \
                        (datetime.datetime.now(pytz.UTC) - order.submitted_at).total_seconds() > 10:
                    reward = self.get_reward()  # calculate reward
                    next_state = self.get_state()  # calculate next state
                    break
        return reward, next_state

    def get_reward(self):
        """
        Calculates the reward from the action taken in the current step.

        Returns:
            float: The reward for the current step.
        """
        # Calculate the soonest stock price - the current stock price
        return (((self.portfolio_value - self.portfolio_values[-20]) / self.portfolio_values[-20]))*1000

    def render(self, reward):
        """
        Renders the environment.
        """
        print(f"Portfolio value: {self.portfolio_value}, Cash: {self.portfolio_values[0]}, Asset price: {self.asset_price}, Reward: {reward}")

    def start(self):
        """
        Starts the environment.
        """
        self.thread2.start()
        sleep(2)
        self.thread1.start()

    def initialize(self):
        '''
        Waits for the full boot up
        '''
        initialized = False
        while not initialized:
            if len(self.data) == self.window_size:
                initialized = True
            else:
                print("Initialization: ", (len(self.data)/self.window_size)*100, "%")
                time.sleep(1)


if __name__ == '__main__':
    # Create a LiveTrader object
    trader = LiveStockEnvironment('AAPL', 128,30)
