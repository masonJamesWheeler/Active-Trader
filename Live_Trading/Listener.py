import collections
import logging
import os
import warnings
from collections import deque, namedtuple
from random import random, randrange
import alpaca_trade_api as tradeapi
from alpha_vantage.timeseries import TimeSeries
import joblib
import numpy as np
import torch
import torch.optim as optim
from torch import device
from Data.data import get_and_process_data, get_all_months
from Data.Get_Fast_Data import get_most_recent_data2
from Environment.StockEnvironment import ReplayMemory, epsilon_decay
from Training.Utils import Transition, execute_action
from Models.DQN_Agent import update_Q_values
from PostGreSQL.Database import *
from Training.Utils import *
from datetime import datetime
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trading:
    def __init__(self, ticker="AAPL", month="2023-07", interval="1Min"):
        self.api_keys = self.get_api_keys()
        self.ts = TimeSeries(key=self.api_keys[0], output_format='pandas')
        self.ticker = ticker
        self.month = month
        self.interval = interval
        self.table_name = f'ticker_{self.ticker}_data'
        self.api = self.initialize_api()
        self.scaler = self.load_scaler()
        self.replay_memory = self.load_replay_memory()
        self.Q_network, self.target_network, self.optimizer = self.load_networks()
        self.last_hidden_state1, self.last_hidden_state2 = self.Q_network.init_hidden(1)
        self.Q_network.to(DEVICE)
        self.target_network.to(DEVICE)
        self.last_state = self.get_new_state()
        self.last_timestamp = self.last_state[0]
        self.last_price = self.get_price()
        self.last_shares = self.last_state[-1, -1]
        self.action = 0
        self.steps_done = 0

    def get_api_keys(self):
        alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
        paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")
        return alpha_vantage_api_key, paper_alpaca_key, paper_alpaca_secret_key

    def initialize_api(self):
        api = tradeapi.REST(
            self.api_keys[1],
            self.api_keys[2],
            base_url='https://paper-api.alpaca.markets',
            api_version='v2'
        )
        return api

    def load_scaler(self):
        scaler_path = f"../Scalers/{self.ticker}_{self.interval}_scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        return scaler

    def load_replay_memory(self):
        replay_memory = ReplayMemory(50000)
        replay_memory.load_memory(self.ticker)
        return replay_memory

    def load_networks(self):
        architecture = "RNN"
        feature_size = 35
        num_actions = 11
        dropout_rate = 0.2
        dense_layers = 2
        hidden_size = 128
        dense_size = 128
        batch_size = 512

        Q_network = DQN(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
            architecture=architecture,
            dense_layers=dense_layers,
            dense_size=dense_size,
            dropout_rate=dropout_rate
        )

        target_network = DQN(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_actions=num_actions,
            architecture=architecture,
            dense_layers=dense_layers,
            dense_size=dense_size,
            dropout_rate=dropout_rate
        )

        target_network.load_state_dict(Q_network.state_dict())

        Q_network.load_weights(False, self.ticker, Q_network.dense_layers_num, Q_network.dense_size, Q_network.hidden_size,
                               Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

        target_network.load_weights(True, self.ticker, Q_network.dense_layers_num, Q_network.dense_size,
                                    Q_network.hidden_size,
                                    Q_network.dropout_rate, Q_network.input_size, Q_network.num_actions)

        optimizer = optim.Adam(Q_network.parameters())

        # Move the model to the device
        Q_network.to(device)
        target_network.to(device)

        return Q_network, target_network, optimizer

    def perform_trade(self, action, asset_price):
        try:
            self.api.cancel_all_orders()
            self.validate_action(action)
            if action != 0:
                portfolio_value, share_value = self.update_portfolio_values()
                print(f"Portfolio Value: {portfolio_value}")
                print(f"Share Value: {share_value}")
                print(f"Asset Price: {asset_price}")

                out_of_bounds, order = self.execute_trade(action, portfolio_value, share_value, asset_price)
        except Exception as e:
            logging.error(f"Failed to perform trade: {e}")

    def get_price(self):
        try:
            price = float(self.ts.get_quote_endpoint(symbol=self.ticker)[0]['05. price'].iloc[0])
            return price
        except Exception as e:
            logging.error(f"Failed to get price: {e}")

    def validate_action(self, action):
        try:
            if action not in range(11):
                raise ValueError("Action not recognized")
        except ValueError as e:
            logging.error(f"Invalid action value: {e}")

    def execute_trade(self, action, portfolio_value, initial_shares, asset_price):
        try:
            order = None
            desired_portfolio_value = self.get_desired_portfolio_value(action, portfolio_value)
            print(f"Desired Portfolio Value: {desired_portfolio_value}")
            desired_shares = self.get_share_details(asset_price, desired_portfolio_value)
            print(f"Initial Shares: {initial_shares}")
            print(f"Desired Shares: {desired_shares}")

            just_closed_position = False

            # Check if we are trying to switch from long to short position
            if desired_shares < 0 and initial_shares > 0:
                self.api.close_position(self.ticker)  # Close the long position
                just_closed_position = True
                initial_shares = 0  # We don't have any shares now
                print(f"Closed long position before opening short position")
            # Check if we are trying to switch from short to long position
            elif desired_shares > 0 and initial_shares < 0:
                self.api.close_position(self.ticker)  # Close the short position
                just_closed_position = True
                initial_shares = 0  # We don't have any shares now
                print(f"Closed short position before opening long position")

            side, shares_to_trade = self.get_side_and_shares_to_trade(initial_shares, desired_shares)
            print(f"Side: {side}")
            print(f"Shares to Trade: {shares_to_trade}")

            out_of_bounds, order = self.submit_order(side, shares_to_trade, just_closed_position)
            return out_of_bounds, order
        except Exception as e:
            logging.error(f"Failed to execute trade: {e}")

    def get_desired_portfolio_value(self, action, portfolio_value):
        try:
            if action < 6:
                return portfolio_value * action * 0.2
            else:
                return portfolio_value * (action - 5) * -0.05
        except Exception as e:
            logging.error(f"Failed to get desired portfolio value: {e}")

    def get_share_details(self, asset_price, desired_portfolio_value):
        try:
            return int(desired_portfolio_value / asset_price)
        except Exception as e:
            logging.error(f"Failed to get share details: {e}")

    def get_side_and_shares_to_trade(self, initial_shares, desired_shares):
        try:
            if desired_shares < initial_shares:
                return 'sell', initial_shares - desired_shares
            elif desired_shares > initial_shares:
                return 'buy', desired_shares - initial_shares
            else:
                return 'hold', 0
        except Exception as e:
            logging.error(f"Failed to determine trade side and shares: {e}")

    def submit_order(self, side, shares, just_closed_position=False):
        try:
            if side == 'hold' or shares == 0:
                print("No order submitted, holding current position.")
                return False, None

            if just_closed_position:
                time.sleep(1)  # Wait for a second before submitting the new order

            order = self.api.submit_order(symbol=self.ticker, qty=shares, side=side, type='market', time_in_force='day')
            print(f"{side} order for {shares} shares submitted")
            return False, order
        except Exception as e:
            logging.error(f"Order submission failed: {e}")
            return True, None

    def update_portfolio_values(self):
        try:
            account = self.api.get_account()
            try:
                position = self.api.get_position(self.ticker)
            except Exception as e:
                position = None

            if position is not None:
                shares = int(position.qty)
            else:
                shares = 0

            if account is None or not hasattr(account, 'portfolio_value'):
                print("Invalid account data. Exiting...")
                return None, None, None
            return float(account.portfolio_value), shares
        except Exception as e:
            logging.error(f"Failed to update portfolio values: {e}")

    def get_new_state(self):
        try:
            history = np.array(get_latest_n_rows(self.table_name, 127))[:, 1:-2].astype(float)
            new_data = get_most_recent_data2(self.ticker, '1min', scaler=self.scaler)[np.newaxis, :]
            return np.concatenate((history, new_data), axis=0)
        except Exception as e:
            logging.error(f"Failed to get new state: {e}")

    def get_new_state2(self):
        try:
            history = self.scaler.inverse_tranform(np.array(get_latest_n_rows(self.table_name, 127))[:, 1:31].astype(float))
            print(history)
            new_data = self.ts.get_quote_endpoint(self.ticker)[0]
            new_data = np.array([new_data['02. open'], new_data['03. high'],
                              new_data['04. low'], new_data['05. price'],
                                 new_data['06. volume']]).reshape(1,5).astype(float)
        #   unscale the history
        except Exception as e:
            logging.error(f"Failed to get new state: {e}")

    def listener(self, steps_done=0, C=10, epsilon=0.2):
        current_minute = datetime.now().minute
        previous_action = None  # Initial value for previous action
        while True:
            try:
                if datetime.now().minute != current_minute:
                    current_minute = datetime.now().minute
                    steps_done += 1
                    action, hidden_state1, hidden_state2, epsilon = execute_action(self.last_state,
                                                                                   self.last_hidden_state1,
                                                                                   self.last_hidden_state2,
                                                                                   epsilon,
                                                                                   11,
                                                                                   self.Q_network)

                    new_state = self.get_new_state()
                    new_price = self.get_price()

                    # If we have a previous action, then calculate reward, perform trade, and print
                    if previous_action is not None:
                        reward = self.last_shares * (new_price - self.last_price)

                        transition = Transition(state=self.last_state, hidden_state1=self.last_hidden_state1,
                                                hidden_state2=self.last_hidden_state2,
                                                action=previous_action, next_state=new_state, reward=reward,
                                                next_hidden_state1=hidden_state1, next_hidden_state2=hidden_state2)

                        self.replay_memory.push(transition)

                        if len(self.replay_memory) >= 512:
                            transitions = self.replay_memory.sample(512)
                            batch = Transition(*zip(*transitions))
                            update_Q_values(batch, self.Q_network, self.target_network, self.optimizer, "RNN")

                        if steps_done % C == 0:
                            self.target_network.load_state_dict(self.Q_network.state_dict())

                        self.perform_trade(previous_action, new_price)  # Perform trade with previous action
                        print(
                            f"Action: {previous_action.item()}, Reward: {reward}, Portfolio Value: {self.portfolio_value}")
                        add_row_to_table(self.table_name, self.last_state[-1], previous_action.item(), self.last_shares)

                    # Update last states, price, and portfolio values
                    self.last_state = new_state
                    self.last_price = new_price
                    self.portfolio_value, self.last_shares = self.update_portfolio_values()

                    self.last_hidden_state1 = hidden_state1
                    self.last_hidden_state2 = hidden_state2

                    # Now we update the previous action to the current one
                    previous_action = action
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            time.sleep(1)


if __name__ == '__main__':
    try:
        # Instantiate a Trading object with your desired parameters
        trader = Trading(ticker='AAPL', month='2023-07', interval='1Min')

        # # Start the trading process
        # trader.listener()
        trader.get_new_state2()

    except Exception as e:
        print(f'An error occurred: {e}')

