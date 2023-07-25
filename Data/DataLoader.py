import numpy as np
import pandas as pd

from Data.Data import get_and_process_data


class DataLoader:
    def __init__(self, X, x_size = 128, y_size = 128, window_lookback_size=400, window_lookforward_size=400):
        self.X = X
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
        y_indices = np.clip(y_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Fetch X and Y data
        x_data = self.X[x_indices]
        y_data = self.X[y_indices]

        return x_data, y_data

    def generate_training_data(self):

        X_data = []
        Y_data = []

        for i in range(self.window_lookback_size, len(self.X) - self.window_lookforward_size):
            x_data, y_data = self.get_data(i)
            X_data.append(x_data)
            Y_data.append(y_data)

        return np.array(X_data), np.array(Y_data)


if __name__ == "__main__":
    data, scaled_data, scaler = get_and_process_data('AAPL', '1Min', 128, month='2023-07')

    # Create a data loader
    data_loader = DataLoader(scaled_data)

    # Generate training data
    X_train, Y_train = data_loader.generate_training_data()

    print(X_train.shape)
    print(Y_train.shape)
