import collections
import logging
import threading
import datetime
import time
from collections import deque
import csv

import joblib
import pytz
import numpy as np
import pandas as pd
import torch
from time import sleep

from Models.DQN_Agent import DQN, MetaModel, update_Q_values
from Training.Utils import Transition, execute_action, initialize
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
from Data.data import get_last_data
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
        self.last_data = None
        self.scaler = joblib.load(f"Scalers/{ticker}_{'1min'}_scaler.pkl")
        self.portfolio_values = deque(maxlen=window_size)
        self.alpha_vantage_data = None

        # self.update_indicators()
        self.api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets', api_version='v2')

        self.cash = 0
        self.share_value = 0
        self.account = 0
        self.positionSize = 0
        self.asset_price = 0
        self.portfolio_value = 0

        self.BATCH_SIZE = 512
        self.architecture = "RNN"
        self.hidden_size = 128
        self.dense_size = 256
        self.dense_layers = 2

        self.delay_length = 2
        self.C = 10
        self.Save_every = 1000
        self.steps_done = 0

        self.memoryReplay, self.num_actions, self.Q_network, self.target_network, self.optimizer,\
            self.hidden_state1, self.hidden_state2 = initialize(self.architecture, self.hidden_size,
                                                                self.dense_size, self.dense_layers)
        self.meta_model = MetaModel(2, 10, 1, window_size)
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters())
        self.meta_model.load_weights(2, 10, 1, window_size)

        self.delayed_states = collections.deque(maxlen=self.delay_length)
        self.delayed_actions = collections.deque(maxlen=self.delay_length)
        self.delayed_hidden1 = collections.deque(maxlen=self.delay_length)
        self.delayed_hidden2 = collections.deque(maxlen=self.delay_length)
        self.delayed_x_values = collections.deque(maxlen=self.delay_length)

        self.prev_portfolio_value = None
        self.prev_buy_and_hold_value = None
        self.portfolio_changes = []
        self.buy_and_hold_changes = []

    def alpha_vantage_update(self):
        """Updates the environment's data using the Alpha Vantage API.

        This function attempts to retrieve the most recent stock data for 'AAPL'
        from the Alpha Vantage API up to 10 times. If the most recent quote
        from the API is different from the previously stored data, it updates
        the stored data and transforms the new data to the appropriate format.
        It then retrieves account information, checks its validity, and updates
        the portfolio and account data stored in the environment.

        Returns:
            bool: True if new data was successfully retrieved and processed,
                  False otherwise (e.g., if the most recent quote from the
                  API was not different from the previously stored data,
                  or if an exception occurred).
        """
        try:
            for _ in range(10):
                # Get the latest quote from the Alpha Vantage API
                latest_quote = self.ts.get_intraday(symbol='AAPL', interval='1min', outputsize='full')[0].iloc[0]

                # Check if the latest quote is different from the previously stored data
                if self.last_data is None or not np.array_equal(latest_quote, self.last_data[0]):
                    # If the quote is different, update the stored data
                    self.last_data = latest_quote.copy()

                    # Retrieve the most recent data for 'AAPL'
                    new_data = get_most_recent_data("AAPL", "1min")

                    # Add a new dimension to the data, then scale it and flatten it
                    new_data = np.expand_dims(new_data, axis=0)
                    new_data = self.scaler.transform(new_data)
                    new_data = new_data.flatten()

                    # Get the account information
                    self.account = account = self.api.get_account()

                    # Check if the account data is valid
                    if account is None or not hasattr(account, 'portfolio_value'):
                        print("Invalid account data. Exiting...")
                        return False

                    # Update the portfolio and account data stored in the environment
                    self.portfolio_value = portfolio_value = float(account.portfolio_value)
                    self.portfolio_values.append(portfolio_value)
                    self.share_value = float(account.position_value)
                    self.cash = float(account.cash)
                    self.asset_price = float(latest_quote['4. close'])

                    # Append the new data to the data deque
                    self.data.append(np.concatenate((new_data, np.array([
                        self.cash / self.portfolio_value, self.share_value / self.portfolio_value])), axis=0))

                    return True

                # If the latest quote from the API was not different from the previously stored data, sleep before trying again
                time.sleep(0.3)

            return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

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
        # start a timer to see how long this takes
        start = time.time()
        if self.alpha_vantage_update():
            self._cancel_all_orders()

            if action not in range(11):
                raise ValueError("Action not recognized")

            if action == 0:
                outOfBounds = True
                order = None
            else:
                outOfBounds, order = self._execute_trade(action)

            portfolioValue, next_state = self._compute_portfolio_value_and_next_state(order, outOfBounds)

            self.render()
            print(f"Step took: {time.time() - start} seconds")
            return next_state, portfolioValue, False

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
        if action < 6:
            desired_portfolio_value = self.portfolio_value * action * 0.2
            side = 'buy'
        elif action < 11:
            desired_portfolio_value = self.portfolio_value * (action - 5) * 0.05
            side = 'sell'
        else:
            raise ValueError("Action not recognized")

        # Calculate the desired numbers of shares
        initial_shares = int(self.share_value / self.asset_price)
        desired_shares = int(desired_portfolio_value / self.asset_price)

        try:
            order = self.api.submit_order(symbol=self.ticker, qty=desired_shares - initial_shares, side=side,
                                          type='market', time_in_force='day')
            print(f"{side} order submitted")

        except Exception as e:
            logging.error(f"Order submission failed: {e}")
            return True, None

        return False, order

    def _compute_portfolio_value_and_next_state(self, order, outOfBounds):
        """
        Compute the reward and the next state.
        Args:
            outOfBounds (bool): Whether trading was possible.
        Returns:
            tuple: The reward and the next state.
        """
        if outOfBounds:
            portfolioValue = self.portfolio_value
            next_state = self.get_state()  # calculate next state
        else:
            while True:
                if order.filled_at is not None or \
                        (datetime.datetime.now(pytz.UTC) - order.submitted_at).total_seconds() > 10:
                    portfolioValue = self.portfolio_value
                    next_state = self.get_state()  # calculate next state
                    break
        return portfolioValue, next_state

    def train(self):
        """Trains the stock trading model.

        This method is the primary training loop for the stock trading model.
        In each iteration of the loop, it selects an action based on the current
        state, performs the selected action, and then calculates the reward
        based on the change in portfolio value.

        The method also manages the memory replay buffer, updating the Q-values
        when the buffer is full and periodically updating the target network.

        The training loop continues indefinitely, with a sleep at the end of
        each loop to synchronize the loop with the real-time stock data.
        """
        state = self.get_state()

        while True:
            # Select an action based on the current state
            action, self.hidden_state1, self.hidden_state2 = execute_action(state, self.hidden_state1,
                                                                            self.hidden_state2, self.steps_done,
                                                                            self.num_actions, self.Q_network)

            # Perform the selected action
            next_state, x_value, done = self.perform_trade_step(action)

            # Store the state, action and hidden states in the dequeue
            self.delayed_states.append(state)
            self.delayed_actions.append(action)
            self.delayed_hidden1.append(self.hidden_state1)
            self.delayed_hidden2.append(self.hidden_state2)
            self.delayed_x_values.append(x_value)

            if len(self.delayed_states) == self.delay_length:
                # We now have enough steps to calculate a delayed reward
                delayed_state = self.delayed_states[0]
                delayed_action = self.delayed_actions[0]
                delayed_h1 = self.delayed_hidden1[0]
                delayed_h2 = self.delayed_hidden2[0]
                delayed_x_value = self.delayed_x_values[0]

                # Calculate the reward
                reward = (x_value - delayed_x_value)
                print(reward)

                # Add the transition to the memory replay
                self.memoryReplay.push(
                    (delayed_state, delayed_h1, delayed_h2, delayed_action, state, reward, self.hidden_state1,
                     self.hidden_state2))

            # If the memory replay is full, sample a batch and update the Q-values
            if len(self.memoryReplay) >= self.BATCH_SIZE:
                transitions = self.memoryReplay.sample(self.BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                update_Q_values(batch, self.Q_network, self.target_network, self.optimizer, self.architecture)

            # If the number of steps is a multiple of C, update the target network
            if self.steps_done % self.C == 0:
                self.target_network.load_state_dict(self.Q_network.state_dict())

            # If the number of steps is a multiple of Save_every, save the Q-network and the Target-network
            if self.steps_done % self.Save_every == 0:
                torch.save(self.Q_network.state_dict(), "models/{}.pth".format(self.ticker))
                torch.save(self.target_network.state_dict(), "models/{}_target.pth".format(self.ticker))

            # Update the current state
            state = next_state

            # Sleep until the start of the next minute to synchronize with real-time stock data
            current_time = time.time()
            seconds = current_time % 60  # get the number of seconds past the minute
            sleep_time = 60 - seconds  # calculate the number of seconds until the next minute

            time.sleep(sleep_time)  # sleep until the start of the next minute

    def load_state_deque(self):
        """Loads the initial state deque with historical stock data.

        This method retrieves the last set of data points for the specified
        ticker from the Alpha Vantage API, scales the data, and stores it in
        the environment's state deque. It also retrieves account information
        from the Alpaca API and updates the environment's portfolio and account
        data.

        The method is designed to be run at the start of the training process
        to initialize the state deque with real historical data.
        """

        # Load the ReplayMemory from the ReplayMemoryCache
        self.memoryReplay.load_memory("AAPL")
        # Load the BackTest_Weights for the DQN
        # Retrieve the last set of data points for the specified ticker from the Alpha Vantage API
        last_data, last_scaled_data, _ = get_last_data(self.ticker, '1min', "2023-07", self.window_size)

        # Retrieve account information from the Alpaca API
        account = self.api.get_account()

        # Update the portfolio and account data stored in the environment
        self.portfolio_value = float(account.portfolio_value)
        self.share_value = float(account.long_market_value)
        self.cash = float(account.cash)

        # Scale the data and store it in the state deque
        for i in range(self.window_size):
            self.data.append(np.concatenate(
                (last_scaled_data[i], [self.cash / self.portfolio_value, self.share_value / self.portfolio_value])))

    def render(self):
        """
        Renders the environment.
        """
        print(f"Portfolio value: {self.portfolio_value}, Cash: {self.cash}, Asset price: {self.asset_price}")
