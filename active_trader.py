import random
from collections import deque
from time import sleep
import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.utils import to_categorical
from datetime import datetime
from model_utils import load_model_from_checkpoint, dense_model, lstm_model, conv_model

# Directory Definitions
dense_model_dir = "my_dir/Stock_Trading_dense/trial_0017"
conv_model_dir = "my_dir/Stock_Trading_conv/trial_0016"
lstm_model_dir = "my_dir/Stock_Trading_lstm/trial_0015"

class StockEnv:
    def __init__(self, data, cash, shares):
        # Raw stock data for each trading day: 30 day window and 14 features per day.
        self.data = data  # Shape: [total_days, 30, 14]

        # Initial cash available to the agent.
        self.cash = cash  # Scalar
        # Initial cash available to the agent.
        self.initial_cash = cash

        # The price of the share at the start of the trading period.
        self.initial_share_price = data[0,-1,-3]  # Scalar

        # Initial number of shares that the agent owns.
        self.shares = shares  # Scalar

        # Number of possible actions.
        self.action_space = 21  # Scalar

        # Current trading day index.
        self.current_day = 0  # Scalar

        # Total number of trading days available in the data.
        self.total_days = len(self.data)  # Scalar

        # Whether the trading simulation is done (reached the end of available data).
        self.done = False  # Boolean

        # State representation: proportion of total value in shares, proportion in cash, number of shares, cash.
        # Shape: (4,)
        self.state = self.update_state()  # Shape: (4,)

    def reset(self):
        # Reset the trading environment to its initial state.
        self.cash = 10000  # Scalar
        self.shares = self.data[0, -1, 3] # Scalar
        self.done = False  # Boolean
        self.current_day = 0  # Scalar

        # Reset the state representation.
        self.state = self.update_state()  # Shape: (4,)
        # Return the initial state.
        return self.state

    def update_state(self):
        # Compute the current share price from data
        current_share_price = self.data[self.current_day][-1, 3]  # Scalar

        # Calculate total portfolio value
        portfolio_value = self.shares * current_share_price + self.cash  # Scalar

        # The state now contains the proportions of portfolio value in shares and cash,
        # the number of shares, and the amount of cash. All are scalars.
        return np.array([self.shares * current_share_price / portfolio_value,
                         self.cash / portfolio_value,
                         self.shares,
                         self.cash])  # Shape: (4,)

    def get_total_value(self):
        # Compute the current share price from data
        current_share_price = self.data[self.current_day][-1, 3]  # Scalar

        # Return total portfolio value
        return self.shares * current_share_price + self.cash  # Scalar

    def step(self, action):
        # Compute the current share price from data
        current_share_price = self.data[self.current_day][-1, 3]  # Scalar

        # Calculate the total portfolio value at the beginning of this step.
        initial_total_value = self.get_total_value()  # Scalar

        # Process the action taken by the agent.
        if action == 0:  # Sell all shares.
            self.cash += self.shares * current_share_price
            self.shares = 0
        elif action >= 1 and action <= 10:  # Sell a portion of shares, 10% to 100%.
            sold_shares = self.shares * (action / 10)  # Scalar
            self.cash += sold_shares * current_share_price
            self.shares -= sold_shares
        elif action == 11:  # Hold, i.e., do not change the share or cash holdings.
            pass
        elif action >= 12 and action <= 21:  # Buy more shares, using 10% to 100% of cash.
            bought_shares = self.cash * ((action - 11) / 10) / current_share_price  # Scalar
            self.cash -= bought_shares * current_share_price
            self.shares += bought_shares
        else:
            raise ValueError("Action must be between 0 and 21")

        # Update the state based on the new cash and share holdings.
        self.state = self.update_state()  # Shape: (4,)

        # If we've reached the last day of the data, the simulation is done.
        if self.current_day == self.total_days - 1:
            self.done = True
        else:
            # Otherwise, move on to the next day.
            self.current_day += 1

        # Calculate the reward as the change in total portfolio value due to the action.
        reward = self.get_total_value() - initial_total_value  # Scalar

        # Return the new state, the reward from this step, and whether the simulation is done.
        return self.state, reward, self.done  # Shape of self.state: (4,)

    def render(self):
        # Display the current portfolio value vs if we would have just held the initial shares.
        print("Day: ", self.current_day)
        print("Total portfolio value: ", self.get_total_value(), "Total value if initial shares were held: ", self.initial_share_price * self.shares + self.initial_cash)

    def get_state(self):
        return self.state

    def get_cash(self):
        return self.cash

    def get_current_day(self):
        return self.current_day

class DQNAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.001, memory_size=2000):
        # The shape of the state space.
        self.state_shape = state_shape  # Shape: As specified during object instantiation.

        # The shape of the action space.
        self.action_shape = action_shape  # Shape: As specified during object instantiation.

        # The memory for storing experiences (state, action, reward, next_state, done).
        self.memory = deque(maxlen=memory_size)

        # Discount factor for future rewards.
        self.gamma = 0.95

        # Initial exploration rate.
        self.epsilon = 1.0

        # Minimum exploration rate.
        self.epsilon_min = 0.1

        # Decay rate for the exploration rate.
        self.epsilon_decay = 0.995

        # The DQN model.
        self.model = self.create_model(learning_rate)

        # Models used for predicting intermediate features.
        self.intermediate_models = self.load_models()

    def create_model(self, learning_rate):
        # Define the structure of the DQN model.
        inputs = keras.Input(shape=(self.state_shape, ), name='state')  # Shape: (state_shape,)
        x = keras.layers.Dense(60, activation="relu")(inputs)
        x = keras.layers.Dense(60, activation="relu")(x)
        action_q_vals = keras.layers.Dense(self.action_shape, activation="linear", name='action')(x)  # Shape: (action_shape,)
        model = keras.Model(inputs=inputs, outputs=action_q_vals)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    def load_models(self):
        # Load the models that generate intermediate features.
        # These could be any type of models, depending on the specifics of your implementation.
        dense_model_path = "saved_weights/dense_model.h5"
        conv1d_model_path = "saved_weights/conv1d_model.h5"
        lstm_model_path = "saved_weights/lstm_model.h5"
        transformer_model_path = "saved_weights/transformer_model.h5"

        # Load each of the models


        # Return the models in a list
        return [dense_model, conv1d_model, lstm_model, transformer_model]

    def act(self, data, state):
        # Decide on an action based on the current state and the Q-value estimates.

        # Concatenate the state with the predictions from the intermediate models.
        # Shape: (1, state_shape + sum of shapes of intermediate predictions)

        # If a randomly drawn number is less than epsilon, choose a random action.
        # Otherwise, choose the action with the highest estimated Q-value.

        # Get predictions from each of the intermediate models
        # Get predictions from each of the intermediate models
        # Since each model predicts a shape of 3 and there are four models, we get a shape of (4,3)
        intermediate_predictions = [model.predict(data.reshape(1, *data.shape))[0] for model in
                                    self.intermediate_models]

        # Concatenate the predictions to reshape them into one dimension
        # This will convert the list of 4 arrays of shape (3,) into a single array of shape (12,)
        intermediate_predictions = np.concatenate(intermediate_predictions).reshape(1, -1)

        # Combine the intermediary model predictions with the state
        # If state is of shape (4,), then full_state will be of shape (1,16)
        full_state = np.concatenate([state.reshape(1, -1), intermediate_predictions], axis=1)

        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # With epsilon probability, returns a random action. This is part of the exploration process
            return np.random.choice(self.action_shape)

        # Predicts Q-values using our primary network. Q-values shape will be (1, action_shape)
        q_values = self.model.predict(full_state)

        # Select the action with the highest Q-value. Decay epsilon as long as it's greater than epsilon_min
        action = np.argmax(q_values[0])  # Shape: single integer, corresponding to the chosen action.

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action  # Return the chosen action

    def remember(self, state, action, reward, next_state, done):
        # Store this experience in the memory. Each element in the memory will be a tuple of (state, action, reward, next_state, done)
        # state and next_state are arrays of shape (4,), action is an integer, reward is a float, done is a boolean
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Sample a batch of experiences from memory to train the network
        minibatch = random.sample(self.memory,
                                  batch_size)  # minibatch shape: (batch_size, 5) because each memory is a 5-elements tuple

        for state, action, reward, next_state, done in minibatch:
            # Predict Q-values for the current state
            target = self.model.predict(state)  # Shape: (1, action_shape)

            if done:
                # If the episode is done, there is no future reward, so the Q-value is the current reward
                target[0][
                    action] = reward  # target shape: (1, action_shape), so we are just updating the Q-value for the chosen action
            else:
                # If the episode is not done, update the Q-value of the chosen action using the reward and the max future Q-value
                Q_future = max(self.model.predict(next_state)[0])  # Shape: (1, action_shape), we take the max Q-value
                target[0][action] = reward + Q_future * self.gamma

            # Fit the model to the current state and target Q-values
            self.model.fit(state, target, epochs=1, verbose=0)  # Both state and target have shape: (1, action_shape)

def train_agent(data):
    # Initialization
    env = StockEnv(data, cash=10000, shares=1000)
    agent = DQNAgent(state_shape=16,
                     action_shape=env.action_space)  # state_shape is 4 (env state) + 12 (intermediate models)
    batch_size = 32

    # Main training loop
    for episode in range(1000):  # however many episodes you want to run
        state = env.reset()
        for _ in range(env.total_days):  # one step for each day in the environment
            data = env.data[env.current_day]  # get the data for the current day
            action = agent.act(data, state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # render the environment
            env.render()
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


