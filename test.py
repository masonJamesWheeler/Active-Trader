import os
from time import sleep
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import concurrent.futures
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
    def __init__(self, input_size, output_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else output_dim, output_dim) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, num_layers, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else output_dim, output_dim) for i in range(num_layers)])
        self.linear = nn.Linear(output_dim, output_dim)

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

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch_size, sequence_len, hidden_dim]
        energy = self.v(torch.tanh(self.W(encoder_outputs)))  # shape: [batch_size, sequence_len, 1]
        attention = F.softmax(energy, dim=1)  # shape: [batch_size, sequence_len, 1]
        return attention

class TemporalDecoder(nn.Module):
    def __init__(self, in_features, out_features, covariate_dim, hidden_dim, output_len):
        super(TemporalDecoder, self).__init__()
        self.attention = TemporalAttention(hidden_dim)
        self.residual_block = ResidualBlock(2 * hidden_dim, out_features)  # changed hidden_dim to output_dim
        self.covariate_projection = nn.Linear(covariate_dim, hidden_dim)  # changed out_features to hidden_dim
        self.output_len = output_len

    def forward(self, x, projected_covariates):
        outputs = torch.zeros(x.size(0), self.output_len, 5).to(x.device)  # initialize output tensor
        for t in range(self.output_len):
            attention = self.attention(x)
            context = torch.sum(attention * x, dim=1)  # shape: [batch_size, hidden_dim]
            context = context.unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
            projected_covariate = self.covariate_projection(projected_covariates[:, t, :]).unsqueeze(
                1)  # apply projection
            x = self.residual_block(
                torch.cat((context, projected_covariate), dim=-1))  # shape: [batch_size, 1, hidden_dim]
            outputs[:, t, :] = x.squeeze(1)
        return outputs


class TiDE(nn.Module):
    def __init__(self, input_size, num_encoder_layers, num_decoder_layers, output_dim, projected_dim):
        super(TiDE, self).__init__()
        self.feature_projection = FeatureProjection(input_size, projected_dim)
        self.encoder = Encoder(projected_dim, output_dim, num_encoder_layers)
        self.dense_decoder = Decoder(output_dim, num_decoder_layers, output_dim)
        self.temporal_decoder = TemporalDecoder(output_dim, output_dim, input_size, output_dim, 10)
        self.global_residual_connection = nn.Linear(input_size, output_dim)
        self.global_attention = TemporalAttention(output_dim)  # added this line

    def forward(self, x, covariates):
        projected_x = self.feature_projection(x)
        encoded_x = self.encoder(projected_x)
        decoded_x = self.dense_decoder(encoded_x)
        final_output = self.temporal_decoder(encoded_x, covariates)

        # apply global attention to reduce sequence length
        global_residual = self.global_residual_connection(x)
        attention = self.global_attention(global_residual)
        global_residual = torch.sum(attention * global_residual, dim=1)  # shape: [batch_size, hidden_dim]
        global_residual = global_residual.unsqueeze(1).expand(-1, 10, -1)  # shape: [batch_size, 10, hidden_dim]

        return final_output + global_residual

    def save_weights(self, ticker, epoch, live=True):
        if live:
            dir_path = f'Live_Weights'
        else:
            dir_path = f'BackTest_Weights'

        dir_path = f'{dir_path}/{ticker}'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), f'{dir_path}/TiDE_{epoch}.pth')

    def load_weights(self, ticker, live=True):
        if live:
            dir_path = f'Live_Weights'
        else:
            dir_path = f'BackTest_Weights'
        i = 0
        while os.path.isfile(f'{dir_path}/{ticker}/TiDE_{i}.pth'):
            i += 1
        if i > 0:
            self.load_state_dict(torch.load(f'{dir_path}/{ticker}/TiDE_{i - 1}.pth'))
        else:
            print(f"No weights found for {ticker}. Starting with random weights.")


def create_sequences(data, seq_length, pred_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - pred_length):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0:5]  # only take the first 5 columns
        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.stack(ys)

def get_and_process_data_and_generate_training_data(month):
    data, scaled_data, scaler = get_and_process_data('AAPL', '1Min', 128, month=month)
    data_loader = DataLoader(scaled_data)
    x, y = data_loader.generate_training_data()
    return torch.tensor(x), torch.tensor(y[:, :, 0:5])

def main():
    model = TiDE(input_size=30, num_encoder_layers=2, num_decoder_layers=2, output_dim=5,
                 projected_dim=10)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    ticker = 'AAPL'
    live = False

    model.load_weights(ticker=ticker, live=live)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use mean squared error loss
    criterion = nn.MSELoss()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Number of epochs
    epochs = 100

    if live:
        months = get_all_months(2010, 1, 2023, 6)
    else:
        months = get_all_months(2014, 1, 2022, 1)

    test_month = '2023-07'
    test_data, test_scaled_data, test_scaler = get_and_process_data(ticker, '1Min', 128, month=test_month)
    test_data_loader = DataLoader(test_scaled_data)
    test_x, test_y = test_data_loader.generate_training_data()
    test_y = test_y[:, :, 0:5]

    all_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_and_process_data_and_generate_training_data, month) for month in months}
        for future in concurrent.futures.as_completed(futures):
            all_data.append(future.result())

    # Train the model
    for epoch in range(epochs):
        for x, y in all_data:
            x = x.to(device)
            y = y.to(device)

            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x, x)
            loss_mae = criterion(outputs, y)
            loss = loss_mae

            # Now backpropagate using this total loss
            loss.backward()
            optimizer.step()

            # Evaluate the model
            model.eval()
            test_outputs = model(test_x.to(device), test_x.to(device))
            test_loss_mae = criterion(test_outputs, test_y.to(device))
            test_loss = test_loss_mae
            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch, loss.item(), test_loss.item()))

        # Save the model to the run folder
        model.save_weights(ticker, epoch, live=live)


def model_prediction():
    ticker = 'AAPL'

    model = TiDE(input_size=30, output_dim=128, num_encoder_layers=2, num_decoder_layers=3, projected_dim=128)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    model.load_weights(ticker=ticker, live=False)
    model.eval()
    test_month = '2023-07'
    test_data, test_scaled_data, test_scaler = get_and_process_data(ticker, '1Min', 128, month=test_month)

    test_data_loader = DataLoader(test_scaled_data)
    test_x = test_data_loader.get_input_data(int(len(test_scaled_data)-60))

    prediction = model(test_x, test_x)

    prediction = prediction.detach().numpy()

    # fill the rows with zeroes so we go from [129,5] to [129,30]
    unscaled_test_data = test_scaler.inverse_transform(test_scaled_data)
    prediction = np.concatenate((prediction, np.zeros((prediction.shape[0], 25))), axis=1)

    prediction = test_scaler.inverse_transform(prediction)

    fix, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(unscaled_test_data[-200:, 0], label='Actual')
    ax[0].plot(unscaled_test_data[-200:, 1], label='Actual')
    ax[0].plot(unscaled_test_data[-200:, 2], label='Actual')
    ax[0].plot(unscaled_test_data[-200:, 3], label='Actual')
    ax[1].plot(prediction[-200:, 0], label='Open')
    ax[1].plot(prediction[-200:, 1], label='Low')
    ax[1].plot(prediction[-200:, 2], label='High')
    ax[1].plot(prediction[-200:, 3], label='Close')
    ax[0].legend()
    ax[1].legend()
    plt.show()

def model_visual():
    ticker = 'AAPL'

    model = TiDE(input_size=30, num_encoder_layers=2, num_decoder_layers=2, output_dim=5, projected_dim=10)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    model.load_weights(ticker=ticker, live=False)
    model.eval()
    test_month = '2023-07'
    test_data, test_scaled_data, test_scaler = get_and_process_data(ticker, '1Min', 128, month=test_month)

    index = len(test_scaled_data) - 450

    test_data_loader = DataLoader(test_scaled_data)
    test_x, test_y = test_data_loader.get_data(index)
    test_x2 = test_data_loader.get_input_data(index)

    test_x2 = torch.tensor(np.expand_dims(test_x2, axis=0))

    test_y2 = model(test_x2, test_x2).detach().numpy()

    fig, ax = plt.subplots(4, 1, figsize=(15, 10))
    ax[0].plot(test_x[:, 0], label='Open')
    ax[0].plot(test_x[:, 1], label='Low')
    ax[0].plot(test_x[:, 2], label='High')
    ax[0].plot(test_x[:, 3], label='Close')

    ax[1].plot(test_x2[-1, :, 0], label='Open')
    ax[1].plot(test_x2[-1, :, 1], label='Low')
    ax[1].plot(test_x2[-1, :, 2], label='High')
    ax[1].plot(test_x2[-1, :, 3], label='Close')

    ax[2].plot(test_y[:, 0], label='Open')
    ax[2].plot(test_y[:, 1], label='Low')
    ax[2].plot(test_y[:, 2], label='High')
    ax[2].plot(test_y[:, 3], label='Close')

    ax[3].plot(test_y2[-1, :, 0], label='Open')
    ax[3].plot(test_y2[-1, :, 1], label='Low')
    ax[3].plot(test_y2[-1, :, 2], label='High')
    ax[3].plot(test_y2[-1, :, 3], label='Close')

    plt.show()


if __name__ == '__main__':
    # main()
    # model_prediction()
    model_visual()

