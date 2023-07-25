import os
from time import sleep

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Data.Data import get_and_process_data, get_all_data, get_all_months
from Data.DataLoader import DataLoader
from Data.Get_Fast_Data import get_most_recent_data, get_most_recent_data2
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load environment variables from .env file

load_dotenv()
# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas')

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features)  # new shortcut connection
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        residual = self.shortcut(x)  # apply the shortcut connection to the input
        out = self.relu(self.linear1(x))
        out = self.dropout(self.linear2(out))
        out += residual
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.linear(x)

class FeatureProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class TemporalDecoder(nn.Module):
    def __init__(self, in_features, out_features, covariate_dim):
        super(TemporalDecoder, self).__init__()
        self.residual_block = ResidualBlock(in_features + out_features, out_features)  # adjust in_features
        self.covariate_projection = nn.Linear(covariate_dim, out_features)

    def forward(self, x, projected_covariates):
        projected_covariates = self.covariate_projection(projected_covariates)
        return self.residual_block(torch.cat((x, projected_covariates), dim=-1))


class TiDE(nn.Module):
    def __init__(self, input_size, hidden_size, num_encoder_layers, num_decoder_layers, output_dim, projected_dim):
        super(TiDE, self).__init__()
        self.feature_projection = FeatureProjection(input_size, projected_dim)
        self.encoder = Encoder(projected_dim, hidden_size, num_encoder_layers)
        self.dense_decoder = Decoder(hidden_size, hidden_size, num_decoder_layers, output_dim)
        self.temporal_decoder = TemporalDecoder(output_dim, 1, input_size)  # pass in the input size as the covariate dimension
        self.global_residual_connection = nn.Linear(input_size, output_dim)

    def forward(self, x, covariates):
        projected_x = self.feature_projection(x)
        encoded_x = self.encoder(projected_x)
        decoded_x = self.dense_decoder(encoded_x)
        final_output = self.temporal_decoder(decoded_x, covariates)
        global_residual = self.global_residual_connection(x)
        return final_output + global_residual

    def save_weights(self):
        torch.save(self.state_dict(), 'TiDE.pth')

    def load_weights(self):
        self.load_state_dict(torch.load('weights/TiDE2.pth'))


def create_sequences(data, seq_length, pred_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - pred_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0:5]  # only take the first 5 columns
        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.stack(ys)


def main():
    # Initialize the model
    model = TiDE(input_size=30, hidden_size=50, num_encoder_layers=2, num_decoder_layers=2, output_dim=5, projected_dim=128)

    # model.load_weights()
    model.eval()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    # Use mean squared error loss
    criterion = torch.nn.MSELoss()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    epochs = 100

    months = get_all_months(2000, 1, 2022, 1)  # train on data up to June 2023

    # Prepare the test data
    test_month = '2023-07'
    test_data, test_scaled_data, test_scaler = get_and_process_data('AAPL', '1Min', 128, month=test_month)

    test_data_loader = DataLoader(test_scaled_data)
    test_x, test_y = test_data_loader.generate_training_data()
    test_y = test_y[:, :, 0:5]

    # Train the model
    for epoch in range(epochs):
        for month in months:
            data, scaled_data, scaler = get_and_process_data('AAPL', '1Min', 128, month=month)
            data_loader = DataLoader(scaled_data)

            x, y = data_loader.generate_training_data()

            # only keep the first 5 columns
            y = y[:, :, 0:5]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, x)  # use x as both the input and the covariates
            loss = criterion(outputs, y)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Evaluate the model
            model.eval()
            test_outputs = model(test_x, test_x)
            test_loss = criterion(test_outputs, test_y)
            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch, loss.item(), test_loss.item()))

    #     Save the model to the run folder
        torch.save(model.state_dict(), 'Weights/TiDE{}.pth'.format(epoch))

def model_prediction_test():
    model = TiDE(input_size=30, hidden_size=50, num_encoder_layers=2, num_decoder_layers=2, output_dim=5, projected_dim=128)
    model.load_weights()
    model.eval()

    test_month = '2023-07'
    test_data, test_scaled_data, test_scaler = get_and_process_data('AAPL', '1Min', 128, month=test_month)
    test_data_loader = DataLoader(test_scaled_data)
    test_x = test_data_loader.get_input_data(len(test_scaled_data)-1)
    print(test_x.shape)

    prediction = model(test_x, test_x)
    print(prediction.shape)
#   convert torch to numpy
    prediction = prediction.detach().numpy()
    plt.figure(figsize=(20, 10))
    plt.plot(prediction[:, 0], label='Open')
    plt.plot(prediction[:, 1], label='High')
    plt.plot(prediction[:, 2], label='Low')
    plt.plot(prediction[:, 3], label='Close')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()