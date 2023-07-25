import numpy as np
import pandas as pd
class DataLoader:
    def __init__(self, X, Y, x_size, y_size, window_lookback_size, window_lookforward_size):
        self.X = X
        self.Y = Y
        self.x_size = x_size
        self.y_size = y_size
        self.window_lookback_size = window_lookback_size
        self.window_lookforward_size = window_lookforward_size

    def get_data(self, index):
        # Ensure index is within bounds
        assert self.window_lookback_size <= index < len(self.X) - self.window_lookforward_size, "Index is out of bounds"

        # Calculate exponential step sizes for X and Y
        x_steps = np.exp(np.linspace(0, np.log(self.window_lookback_size), self.x_size))[::-1]  # reversed
        y_steps = np.exp(np.linspace(0, np.log(self.window_lookforward_size), self.y_size))

        # Get X indices
        x_indices = np.array([index - int(step) for step in x_steps])
        x_indices = np.clip(x_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Get Y indices
        y_indices = np.array([index + int(step) for step in y_steps])
        y_indices = np.clip(y_indices, 0, len(self.Y) - 1)  # ensure indices are within bounds

        # Fetch X and Y data
        x_data = self.X[x_indices]
        y_data = self.Y[y_indices]

        return x_data, y_data

if __name__ == "__main__":
    # Create some dummy data
    X = np.arange(1000)
    Y = np.arange(1000)

    # Create a data loader
    data_loader = DataLoader(X, Y, 128, 30, 400, 400)

    # Get some data
    x_data, y_data = data_loader.get_data(500)
    print(x_data)
    print(y_data)