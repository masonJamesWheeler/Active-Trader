import numpy as np
import pandas as pd
import torch


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

        # Fetch X and Y data and convert to tensors
        x_data = self.X[x_indices]
        y_data = self.X[y_indices]

        return x_data, y_data

    def get_input_data(self, index):
        # Ensure index is within bounds, no lookforward needed
        assert self.window_lookback_size <= index, f'Index is out of bounds, Lookback Window Size: {self.window_lookback_size}, Index: {index}'
        assert index < len(self.X), f'Index is out of bounds, Index: {index}, Length of X: {len(self.X)}'

        # Calculate exponential step sizes for X
        x_steps = np.exp(np.linspace(0, np.log(self.window_lookback_size), self.x_size))[::-1]  # reversed

        # Get X indices
        x_indices = np.array([index - int(step) for step in x_steps])
        x_indices = np.clip(x_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Ensure current index is included
        if index not in x_indices:
            x_indices = np.append(x_indices, index)

        # Fetch X data and convert to tensors
        x_data = self.X[x_indices]

        return x_data

    def generate_training_data(self):
        X_data = []
        Y_data = []

        for i in range(self.window_lookback_size, len(self.X) - self.window_lookforward_size):
            try:
                x_data, y_data = self.get_data(i)
                X_data.append(x_data)
                Y_data.append(y_data)
            except IndexError:
                continue

        # Convert list of tensors to a single tensor
        return torch.stack(X_data), torch.stack(Y_data)



if __name__ == '__main__':
    pass

