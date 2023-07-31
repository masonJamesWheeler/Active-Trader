import numpy as np
import pandas as pd
import torch


class DataLoader:
    def __init__(self, X, x_size=128, y_size=16):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.x_size = x_size
        self.y_size = y_size

    def get_data(self, index):
        # Ensure index is within bounds
        assert index < len(self.X) - self.y_size, "Index is out of bounds"

        # Get X indices
        x_indices = torch.arange(index - self.x_size + 1, index + 1)
        x_indices = torch.clamp(x_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Get Y indices
        y_indices = torch.arange(index + 1, index + self.y_size + 1)
        y_indices = torch.clamp(y_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Fetch X and Y data
        x_data = self.X[x_indices]
        y_data = self.X[y_indices]

        return x_data, y_data

    def get_input_data(self, index):
        # Ensure index is within bounds
        assert index - self.x_size + 1 >= 0, f'Index is out of bounds, X Size: {self.x_size}, Index: {index}'
        assert index < len(self.X), f'Index is out of bounds, Index: {index}, Length of X: {len(self.X)}'

        # Get X indices
        x_indices = torch.arange(index - self.x_size + 1, index + 1)
        x_indices = torch.clamp(x_indices, 0, len(self.X) - 1)  # ensure indices are within bounds

        # Fetch X data
        x_data = self.X[x_indices]

        return x_data

    def generate_training_data(self):
        X_data = []
        Y_data = []

        for i in range(self.x_size, len(self.X) - self.y_size):
                x_data, y_data = self.get_data(i)
                X_data.append(x_data)
                Y_data.append(y_data)

        X_data = torch.tensor(np.array(X_data))
        Y_data = torch.tensor(np.array(Y_data))

        return X_data, Y_data


if __name__ == '__main__':
    data = np.linspace(0, 100, 1000)
    dataloader = DataLoader(data)
    x_data, y_data = dataloader.get_data(500)
    print(x_data.shape)
    print(y_data.shape)