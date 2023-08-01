import os
from time import sleep
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import concurrent.futures
from Data.data import get_and_process_data, get_all_data, get_all_months
from Data.DataLoader import DataLoader
from Data.Get_Fast_Data import get_most_recent_data, get_most_recent_data2
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig

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
        self.linear1 = nn.Linear(in_features, out_features) # Initial linear layer to map input to specified out_features.
        self.linear2 = nn.Linear(out_features, out_features) # Another linear layer to further process the input.
        self.shortcut = nn.Linear(in_features, out_features) # Shortcut connection to allow gradients to flow back easily.
        self.relu = nn.ReLU() # Activation function to introduce non-linearity.
        self.dropout = nn.Dropout(0.5) # Dropout for regularization to prevent overfitting.
        self.layer_norm = nn.LayerNorm(out_features) # Layer normalization to stabilize the learning process.

    def forward(self, x):
        residual = self.shortcut(x) # Apply the shortcut connection.
        out = self.relu(self.linear1(x)) # Apply ReLU after first linear layer.
        out = self.dropout(self.linear2(out)) # Apply dropout after second linear layer.
        out += residual # Add the original input (residual) to the output of the layers.
        out = self.layer_norm(out) # Apply layer normalization.
        return out # Return the output.


class Encoder(nn.Module):
    def __init__(self, intermediate_features, output_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(intermediate_features if i==0 else output_dim, output_dim) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, num_layers, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else output_dim, output_dim) for i in range(num_layers)]) # Decoder is also composed of a sequence of ResidualBlocks.
        self.linear = nn.Linear(output_dim, output_dim) # Final linear layer to map the output to the desired output dimension.

    def forward(self, x):
        for layer in self.layers: # Pass the input through each layer.
            x = layer(x)
        return self.linear(x) # Apply the final linear layer and return the output.

class FeatureProjection(nn.Module):
    def __init__(self, in_features, intermediate_features):
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(in_features, intermediate_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x)) # Apply ReLU after linear transformation

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False) # Linear layer to transform the input before applying attention.
        self.v = nn.Linear(hidden_dim, 1, bias=False) # Linear layer to calculate attention scores.

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch_size, sequence_len, hidden_dim]
        energy = self.v(torch.tanh(self.W(encoder_outputs)))  # Apply transformation and tanh activation, then calculate attention scores.
        attention = F.softmax(energy, dim=1)  # Apply softmax to get attention distribution.
        return attention

class TemporalDecoder(nn.Module):
    def __init__(self, in_features, out_features, covariate_dim, hidden_dim, output_len):
        super(TemporalDecoder, self).__init__()
        self.attention = TemporalAttention(hidden_dim) # Temporal attention mechanism.
        self.residual_block = ResidualBlock(2 * hidden_dim, out_features)  # Residual block for processing inputs.
        self.covariate_projection = nn.Linear(covariate_dim, hidden_dim)  # Linear layer for projecting covariates.
        self.relu = nn.ReLU()  # New ReLU activation
        self.output_len = output_len

    def forward(self, x, projected_covariates):
        outputs = torch.zeros(x.size(0), self.output_len, 5).to(x.device)  # initialize output tensor
        for t in range(self.output_len): # For each time step in the output sequence.
            attention = self.attention(x) # Calculate the attention weights.
            context = torch.sum(attention * x, dim=1)  # Calculate the context vector.
            context = context.unsqueeze(1)  # Add an extra dimension to match the tensor sizes.
            projected_covariate = self.relu(self.covariate_projection(projected_covariates[:, t, :])).unsqueeze(1)  # Apply ReLU after projection
            x = self.residual_block(torch.cat((context, projected_covariate), dim=-1))  # Concatenate context and projected_covariates, then apply residual_block.
            outputs[:, t, :] = x.squeeze(1) # Remove the extra dimension and add the output to the outputs tensor.
        return outputs


class TiDE(nn.Module):
    def __init__(self, input_size, num_encoder_layers, num_decoder_layers, output_dim, intermediate_dim):
        super(TiDE, self).__init__()
        self.feature_projection = FeatureProjection(input_size, intermediate_dim)
        self.encoder = Encoder(intermediate_dim, output_dim, num_encoder_layers)
        self.dense_decoder = Decoder(output_dim, num_decoder_layers, output_dim)
        self.temporal_decoder = TemporalDecoder(output_dim, output_dim, input_size, output_dim, 10)
        self.global_residual_connection = nn.Linear(input_size, output_dim)
        self.global_attention = TemporalAttention(output_dim)
        # Add a non-linearity after the global_residual_connection
        self.global_residual_activation = nn.ReLU()

    def forward(self, x, covariates):
        intermediate_x = self.feature_projection(x)
        encoded_x = self.encoder(intermediate_x)
        decoded_x = self.dense_decoder(encoded_x)
        final_output = self.temporal_decoder(encoded_x, covariates)
        global_residual = self.global_residual_connection(x)
        # Apply the non-linearity to the global_residual
        global_residual = self.global_residual_activation(global_residual)
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


class FinancialPredictor(nn.Module):
    def __init__(self, price_feature_num, time_feature_num, hidden_dim, num_layers, output_dim, output_minutes, dropout_rate=0.2):
        super(FinancialPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_minutes = output_minutes

        self.lstm_price = nn.LSTM(price_feature_num, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)
        self.lstm_time = nn.LSTM(time_feature_num, hidden_dim, num_layers, dropout=dropout_rate, batch_first=True)

        # Transformation LSTM to change sequence length
        self.lstm_transform = nn.LSTM(hidden_dim * 2, hidden_dim * 2, num_layers, dropout=dropout_rate, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Non-linearity (ReLU)
        self.relu = nn.ReLU()

    def forward(self, price_data, time_data):
        h0_price = torch.zeros(self.num_layers, price_data.size(0), self.hidden_dim).to(price_data.device)
        c0_price = torch.zeros(self.num_layers, price_data.size(0), self.hidden_dim).to(price_data.device)

        h0_time = torch.zeros(self.num_layers, time_data.size(0), self.hidden_dim).to(time_data.device)
        c0_time = torch.zeros(self.num_layers, time_data.size(0), self.hidden_dim).to(time_data.device)

        out_price, _ = self.lstm_price(price_data, (h0_price, c0_price))
        out_time, _ = self.lstm_time(time_data, (h0_time, c0_time))

        # Apply Dropout after LSTM
        out_price = self.dropout(out_price)
        out_time = self.dropout(out_time)

        # Concatenate the outputs
        out = torch.cat((out_price, out_time), dim=2)

        # Transformation LSTM to change sequence length
        h0_transform = torch.zeros(self.num_layers, out.size(0), self.hidden_dim * 2).to(out.device)
        c0_transform = torch.zeros(self.num_layers, out.size(0), self.hidden_dim * 2).to(out.device)
        out, _ = self.lstm_transform(out, (h0_transform, c0_transform))

        # Slice the sequence to desired output length
        out = out[:, :self.output_minutes, :]

        # Apply Fully Connected Layers with ReLU activations and Dropout
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)

        return out

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
    data, scaled_data, time_features, scaler = get_and_process_data('AAPL', '1Min', 128, month=month)
    data_loader = DataLoader(scaled_data)
    date_data_loader = DataLoader(time_features)
    x, y = data_loader.generate_training_data()
    time_x, time_y = date_data_loader.generate_training_data()

    return (torch.tensor(x).float(), torch.tensor(y[:, :, 0:5]).float(),
            torch.tensor(time_x).float(), torch.tensor(time_y).float())


def main():
    ticker = 'AAPL'
    live = False

    model = FinancialPredictor(price_feature_num=30, time_feature_num=5, hidden_dim=16, num_layers=2,
                               output_dim=5, output_minutes=16)

    model.load_weights(ticker, live=live)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Use the Mean Squared Error loss function
    criterion = nn.MSELoss()

    # Number of epochs
    epochs = 100

    if live:
        months = get_all_months(2010, 1, 2023, 6)
    else:
        months = get_all_months(2016, 1, 2022, 1)

    all_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_and_process_data_and_generate_training_data, month) for month in months}
        for future in concurrent.futures.as_completed(futures):
            all_data.append(future.result())


    test_month = '2022-03'
    test_data, test_scaled_data, date_features, test_scaler = get_and_process_data(ticker, '1Min', 128, month=test_month)

    test_data_loader = DataLoader(test_scaled_data)
    test_date_data_loader = DataLoader(date_features)
    test_x, test_y = test_data_loader.generate_training_data()
    test_time_x, test_time_y = test_date_data_loader.generate_training_data()

    test_x = test_x.float()
    test_y = test_y[:, :, 0:5].float()
    test_time_x = test_time_x.float()

    best_loss = 1000000

    # Train the model
    for epoch in range(epochs):
        for x, y, time_x, time_y in all_data:
            x = x.to(device) # X is shaped [batch_size, 128, 30]
            y = y.to(device) # Y is shaped [batch_size, 128, 5]
            time_x = time_x.to(device) # X is shaped [batch_size, 128, 5]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x, time_x)
            y_test = model(test_x, test_time_x)

            loss = criterion(y_pred, y)
            test_loss = criterion(y_test, test_y)

            # Now backpropagate using this total loss
            loss.backward()
            optimizer.step()

            print('Epoch: {}, Loss: {}, Test Loss: {}'.format(epoch, loss.item(), test_loss.item()))

            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                model.save_weights(ticker=ticker, live=live, epoch=epoch)

            # Clear the stack
            torch.cuda.empty_cache()

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
    test_month = '2022-03'
    test_data, test_scaled_data, date_features, test_scaler = get_and_process_data(ticker, '1Min', 128,
                                                                                   month=test_month)

    test_data_loader = DataLoader(test_scaled_data)
    test_date_data_loader = DataLoader(date_features)
    test_x, test_y = test_data_loader.generate_training_data()
    test_time_x, test_time_y = test_date_data_loader.generate_training_data()

    test_x = test_x.float()
    test_y = test_y[:, :, 0:5].float()
    test_time_x = test_time_x.float()

    test_x = torch.tensor(test_x[-1]).unsqueeze(0)
    test_time_x = torch.tensor(test_time_x[-1]).unsqueeze(0)
    test_y = torch.tensor(test_y[-1]).unsqueeze(0)

    model = FinancialPredictor(price_feature_num=30, time_feature_num=5, hidden_dim=16, num_layers=2,
                                 output_dim=5, output_minutes=16)

    model.load_weights(ticker=ticker, live=False)
    model.eval()

    prediction = model(test_x, test_time_x)

    prediction = prediction.detach().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(test_y[0, :, 0], label='Actual')
    ax[0].plot(test_y[0, :, 1], label='Actual')
    ax[0].plot(test_y[0, :, 2], label='Actual')
    ax[0].plot(test_y[0, :, 3], label='Actual')
    ax[1].plot(prediction[0, :, 0], label='Open')
    ax[1].plot(prediction[0, :, 1], label='Low')
    ax[1].plot(prediction[0, :, 2], label='High')
    ax[1].plot(prediction[0, :, 3], label='Close')
    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    main()
    # model_prediction()
    model_visual()