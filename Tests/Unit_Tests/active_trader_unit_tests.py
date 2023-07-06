import unittest
import numpy as np
from collections import deque
from tensorflow import keras

from Environment.StockEnvironment import StockEnvironment


class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        """Set up a trading environment and a DQN agent before each test."""
        self.data = np.random.rand(100, 30, 14)  # Random stock data
        self.cash = 1000  # Initial cash
        self.shares = 10  # Initial shares
        self.env = StockEnvironment(self.data, self.cash, self.shares)  # Environment
        self.agent = DQNAgent(4, 21)  # DQN Agent

    def test_environment_init(self):
        """Test whether the environment is initialized correctly."""
        # Check if env is an instance of StockEnv
        self.assertIsInstance(self.env, StockEnv)

        # Check if env.data is a numpy array and has the correct shape
        self.assertIsInstance(self.env.data, np.ndarray)
        self.assertEqual(self.env.data.shape, (100, 30, 14))

        # Check if cash, shares are integers or floats
        self.assertIsInstance(self.env.cash, (int, float))
        self.assertIsInstance(self.env.shares, (int, float))

        # Check initial values of current_day, total_days, done flag, and state shape
        self.assertEqual(self.env.current_day, 0)
        self.assertEqual(self.env.total_days, 100)
        self.assertFalse(self.env.done)
        self.assertIsInstance(self.env.state, np.ndarray)
        self.assertEqual(self.env.state.shape, (4,))

    def test_reset_environment(self):
        """Test whether the environment can be reset correctly."""
        initial_state = self.env.reset()

        # After reset, current_day should be 0 and done should be False
        self.assertEqual(self.env.current_day, 0)
        self.assertFalse(self.env.done)

        # Check if the cash and shares are reset to initial values
        self.assertEqual(self.env.cash, 10000)
        self.assertEqual(self.env.shares, self.data[0, -1, 3])

        # Check if the shape of state is correct
        self.assertEqual(self.env.state.shape, (4,))
        self.assertIsInstance(initial_state, np.ndarray)
        self.assertEqual(initial_state.shape, (4,))

    def test_environment_step(self):
        """Test the step function of the environment."""
        initial_state = self.env.get_state()
        new_state, reward, done = self.env.step(5)  # Perform an action

        # Check if new_state, reward and done have the correct types and shape
        self.assertIsInstance(new_state, np.ndarray)
        self.assertEqual(new_state.shape, (4,))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)

    def test_agent_init(self):
        """Test whether the agent is initialized correctly."""
        # Check if agent is an instance of DQNAgent
        self.assertIsInstance(self.agent, DQNAgent)

        # Check if state_shape and action_shape are integers
        self.assertIsInstance(self.agent.state_shape, int)
        self.assertIsInstance(self.agent.action_shape, int)

        # Check if memory is a deque
        self.assertIsInstance(self.agent.memory, deque)

        # Check if gamma, epsilon, epsilon_min, epsilon_decay are floats
        self.assertIsInstance(self.agent.gamma, float)
        self.assertIsInstance(self.agent.epsilon, float)
        self.assertIsInstance(self.agent.epsilon_min, float)
        self.assertIsInstance(self.agent.epsilon_decay, float)

        # Check if model is a keras.Model
        self.assertIsInstance(self.agent.model, keras.Model)

    def test_agent_remember(self):
        """Test whether the agent can remember (store) experiences correctly."""
        initial_length = len(self.agent.memory)  # Length of memory before remembering
        self.agent.remember(np.array([0, 0, 0, 0]), 0, 1, np.array([1, 1, 1, 1]), False)  # Remember an experience

        # Check if memory has grown by 1 after remembering
        self.assertEqual(len(self.agent.memory), initial_length + 1)

if __name__ == '__main__':
    unittest.main()
