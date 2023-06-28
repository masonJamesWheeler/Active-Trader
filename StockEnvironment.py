import math
import random
from collections import deque
from time import sleep

import numpy as np
import pandas as pd
import torch


class StockEnvironment:
    def __init__(self, starting_cash, starting_shares, data, window_size, feature_size, stack_size, price_column):
        self.action_space = list(range(11))  # 11 actions, from 0 to 10
        self.observation_space = torch.zeros((stack_size, window_size, feature_size), dtype=torch.float32).to('cpu') # Modify for your device, i.e., use 'cuda' for GPU

        self.episode_ended = False
        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.data = data
        self.window_size = window_size
        self.feature_size = feature_size
        self.stack_size = stack_size
        self.price_column = price_column
        self.current_step = 0
        self.current_price = 0
        self.current_cash = starting_cash
        self.current_shares = starting_shares
        self.current_portfolio_value = starting_cash
        self.current_portfolio_value_history = []
        self.current_portfolio_value_history.append(starting_cash)
        self.stacked_frames = deque(maxlen=stack_size)
        self.batch_size = 32
        self.current_time_step_value = data[self.current_step]

    def sample_action(self):
        return random.choice(self.action_space)

#       Function to fill the deque with zeroes if we are initializing the environment,
#       or to add a new frame to the deque if we are stepping through the environment

    #       Function to get the current price of the stock
    def get_current_price(self):
        return self.data[self.current_step, -1, self.price_column]

#       Function to get the current portfolio value
    def get_current_portfolio_value(self):
        return self.current_cash + self.current_shares * self.current_price

    def step(self, action):
        initial_portfolio_value = self.get_current_portfolio_value()
    #     Check if the episode is over
        if self.episode_ended:
            return self.reset()
    # Make the Trade by the action
        if action == 0:
            pass
        elif action == 1: # Use 20% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.2)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 2: # Use 40% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.4)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 3: # Use 60% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.6)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 4: # Use 80% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*0.8)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 5: # Use 100% of cash value to buy a share
            if self.current_cash > self.current_price:
    #             find how many shares we can buy
                shares_to_buy = int((self.current_cash*1)/self.current_price)
    #             update cash and shares
                self.current_cash -= shares_to_buy * self.current_price
                self.current_shares += shares_to_buy
            else:
                pass
        elif action == 6: # Sell 20% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.2))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 7: # Sell 40% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.4))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 8: # Sell 60% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.6))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 9: # Sell 80% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*0.8))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        elif action == 10: # Sell 100% of shares
            if self.current_shares > 0:
    #             find how many shares we can sell
                shares_to_sell = int((self.current_shares*1))
    #             update cash and shares
                self.current_cash += shares_to_sell * self.current_price
                self.current_shares -= shares_to_sell
            else:
                pass
        else:
            raise ValueError("Action not recognized")

        self.current_step += 1
        self.current_portfolio_value_history.append(self.get_current_portfolio_value())

        done = False
        if self.current_step >= len(self.data):
            done = True
        else:
            self.update_state()
        # calculate the reward as the squared difference between the new portfolio value and the old one while keeping the sign
        sign = 1 if self.get_current_portfolio_value() >= initial_portfolio_value else -1
        reward = ((self.get_current_portfolio_value() - initial_portfolio_value)**2) * sign
        new_state = torch.tensor(self.get_stacked_frames(), dtype=torch.float32).to('cpu')
        # done to tensor
        done = torch.tensor(done, dtype=torch.bool).to('cpu')

        return new_state, reward, done

    def reset(self):
        self._episode_ended = False
        self.current_step = 0
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.current_portfolio_value_history = []
        self.current_portfolio_value_history.append(self.starting_cash)
        self.stacked_frames = deque([], maxlen=self.stack_size)
        self.current_price = self.get_current_price()
        return torch.tensor(self.get_stacked_frames(), dtype=torch.float32).to('cpu')

    def get_stacked_frames(self):
        for i in range(self.stack_size):
            index = self.current_step - self.window_size * i
            if index < 0:
                self.stacked_frames.append(torch.zeros(self.data.shape[1], self.data.shape[2]))
            else:
                self.stacked_frames.append(torch.from_numpy(self.data[index]))
        stacked_frames = list(self.stacked_frames)
        # add a dimension to each frame
        return torch.stack(stacked_frames).to('cpu')

    def get_current_time_step(self):
        observation = self.get_stacked_frames()
        return {'observation': observation, 'reward': 0.0, 'discount': 1.0}  # changed from ts.transition to a dict

    def render(self):
        # Check if step is inside the length of the dataset
        if self.current_step <= len(self.data) -1:
            # You may want to visualize some information
            print(f'Step: {self.current_step}, Portfolio Value: {self.current_portfolio_value}, vs. Buy and Hold: {self.get_buy_and_hold_portfolio_value()}')
        else:
            print('End of dataset')

    def update_state(self):
        self.current_price = self.get_current_price()
        self.current_portfolio_value = self.get_current_portfolio_value()
        self.current_portfolio_value_history.append(self.current_portfolio_value)

    def get_buy_and_hold_portfolio_value(self):
        return self.starting_cash + self.starting_shares * self.data[self.current_step, -1, 3]

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)












