import concurrent.futures
from datetime import datetime

import joblib
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.optim.lr_scheduler import StepLR

from Data.Indicators import *
from Data.data import *
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
alpaca_key = os.getenv("ALPACA_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

scaler = MinMaxScaler(feature_range=(0, 1))
ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas')

import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.2)  # Added new dropout layer

        # Dense layers
        self.fc1 = nn.Linear(64 * 120, 512)  # Increased output size
        self.fc2 = nn.Linear(512, 256)  # Added intermediate dense layer
        self.fc3 = nn.Linear(256, 128)  # Added intermediate dense layer
        self.fc4 = nn.Linear(128, 32)   # Added intermediate dense layer
        self.fc5 = nn.Linear(32, 1)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(120)
        self.batch_norm2 = nn.BatchNorm1d(120)
        self.batch_norm3 = nn.BatchNorm1d(120)
        self.batch_norm4 = nn.BatchNorm1d(512)  # Added for fc1
        self.batch_norm5 = nn.BatchNorm1d(256)  # Added for fc2
        self.batch_norm6 = nn.BatchNorm1d(128)  # Added for fc3

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        # Flatten the LSTM output for the dense layer
        x = x.contiguous().view(x.size(0), -1)

        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = nn.ReLU()(x)
        x = self.dropout4(x)

        x = self.fc2(x)
        x = self.batch_norm5(x)
        x = nn.ReLU()(x)

        x = self.fc3(x)
        x = self.batch_norm6(x)
        x = nn.ReLU()(x)

        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.dropout5(x)

        x = self.fc5(x)
        return torch.sigmoid(x)  # Assuming binary classification



def get_stock_data(symbol, interval, month = '2023-07'):
    stock_df, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize="full", month=month)
    # remove the numbers from the column names, i.e 1. open -> open
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = [col.split(' ')[1] for col in stock_df.columns]
    # add the interval to the column names, i.e open -> open_1min
    # stock_df.columns = [f'{col}_{interval}' for col in stock_df.columns]
    stock_df = stock_df.iloc[::-1]

    return stock_df

def timestamp_to_features(timestamp):

    timestamp = timestamp.to_pydatetime()

    # Define the starting point
    start = datetime(2000, 1, 1)

    # Calculate the difference between the timestamp and the starting point
    difference = timestamp - start

    # Convert the difference to minutes
    minutes_since_start = difference.total_seconds() / 60

    # Cyclical encoding for the hour and minute
    hour_sin = np.sin(2 * np.pi * timestamp.hour/24)
    hour_cos = np.cos(2 * np.pi * timestamp.hour/24)
    minute_sin = np.sin(2 * np.pi * timestamp.minute/60)
    minute_cos = np.cos(2 * np.pi * timestamp.minute/60)

    return [minutes_since_start, hour_sin, hour_cos, minute_sin, minute_cos]

def get_all_data(symbol, interval, window_size, month="2003-01"):
    # Assuming that get_stock_data is a function that gets the OHLCV data
    data = get_stock_data(symbol, interval, month=month)

    if interval in ['1min', '5min', '15min']:
        # Replace the API calls with the function calls
        data['smawindow'] = SMA(data, window_size)
        data['emawindow'] = EMA(data, window_size)
        data['sma100'] = SMA(data, 100)
        data['ema100'] = EMA(data, 100)
        data['sma200'] = SMA(data, 200)
        data['ema200'] = EMA(data, 200)
        data['vwap'] = VWAP(data)
        data['rsi'] = RSI(data, 60)
        macd_line, signal_line, histogram = MACD(data)
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['macd_hist'] = histogram
        upper_band, middle_band, lower_band = Bollinger_Bands(data, 60)
        data['bbands_upper'] = upper_band
        data['bbands_middle'] = middle_band
        data['bbands_lower'] = lower_band
        data['wma'] = WMA(data, window_size)
        data['cci'] = CCI(data, window_size)
        aroon_up, aroon_down = Aroon(data, window_size)
        data['aroon_up'] = aroon_up
        data['aroon_down'] = aroon_down
        data['obv'] = OBV(data)
        fastk, fastd = Stochastic(data)
        data['stoch_slowk'] = fastk
        data['stoch_slowd'] = fastd
        fastk, fastd = Fast_Stochastic(data)
        data['stochf_fastk'] = fastk
        data['stochf_fastd'] = fastd
        fastk, fastd = Stochastic_RSI(data, 60)
        data['stochrsi_fastk'] = fastk
        data['stochrsi_fastd'] = fastd

    # Convert the index to datetime if it's not already
    data.index = pd.to_datetime(data.index)

    # Define the start and end of regular trading hours (in hours)
    start_of_trading = 9.5  # 9:30 AM
    end_of_trading = 16  # 4:00 PM

    # Extract the hour from the index
    hour = data.index.hour + data.index.minute / 60.0

    # Filter the DataFrame to include only rows that fall within regular trading hours
    data = data[(hour >= start_of_trading) & (hour < end_of_trading)]

    # Drop the initial rows that have NaN values due to the rolling window calculations
    data.dropna(inplace=True)

    # Rename columns to include interval information
    data.columns = [f"{col}_{interval}" for col in data.columns]

    return data

def create_windows(data, length):
    '''
    Creates a sliding window of length `length` over the data. The window is shifted by 1 each time.
    We should only lose `length` rows from the start of the data.
    '''
    # check if the data is a dataframe
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    windows = []
    for i in range(len(data) - length):
        windows.append(data[i:i + length])
    windows = np.array(windows)

    assert len(windows) == len(data) - length
    return windows

def prepare_labels(data, lookforward_length = 120, threshold = 0.01):
    '''
    Creates a binary label for each window. The label is 1 if the price increases by more than `threshold` over the next `lookforward_length` minutes.
    '''
    # check if the data is a dataframe
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    labels = []
    for i in range(len(data) - lookforward_length):
        # get the price at the start of the window
        start_price = data[i, -1, 3]
        # get the price `lookforward_length` minutes later
        end_price = data[i + lookforward_length, -1, 3]
        # calculate the percentage change
        pct_change = (end_price - start_price) / start_price
        # if the price increases by more than `threshold`, label as 1
        if pct_change > threshold:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)

    assert len(labels) == len(data) - lookforward_length
    # remove the first `lookforward_length` rows from the data
    data = data[lookforward_length:]
    # assert that the lengths of the data and labels are the same
    assert len(data) == len(labels)
    return data, labels

def display_results(x, predictions):
    '''
    Display the results of the predictions, the price for any given step is at [-1,3] in the window.
    and its prediction is singular and 1 if it predicts an increase and 0 if not.

    We are going to plot a green dot on the given stock price if the prediction is 1.
    '''

    # convert to numpy array if it's a tensor
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()

    # Creating the plots
    fig, ax = plt.subplots(2, 1, figsize=(20, 15)) # 2 rows, 1 column

    # Plotting the Actual price on the first subplot
    ax[0].plot(x[:, -1, 3], label='Actual', color='b')
    ax[0].set_title('Actual Prices')
    ax[0].legend()
    ax[0].grid(True) # Add gridlines

    # Plotting the Predicted if Increase on the second subplot
    ax[1].scatter(np.arange(len(x)), predictions, c=np.where(predictions==1, 'g', 'r'), label='Predicted if Increase')
    ax[1].set_title('Predicted Increase Probability')
    ax[1].legend()
    ax[1].grid(True) # Add gridlines

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    intervals = ['1min', '5min', '15min', '30min', '60min']

    # Get ALL data (including indicators) for each interval and store in a list
    dfs = [get_all_data('AAPL', interval, window_size=60) for interval in intervals]

    # Combine all dataframes using outer join to align by datetime index
    combined_df = pd.concat(dfs, axis=1, join='outer')

    # Optionally, fill NaN values if required
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.dropna(inplace=True)

    windowed_data = create_windows(combined_df, length=120)

    model = StockPredictor()

    # Binary cross entropy loss
    criterion = nn.BCELoss()

    # Adam optimizer
    # Adam optimizer with L2 regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # StepLR scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR by half every 5 epochs

    # Example training loop
    for epoch in range(10):
        iter = 0
        for month in get_all_months(start_year=2000, start_month=1, end_year=2023, end_month=7):
            iter += 1

            intervals = ['1min', '5min', '15min', '30min', '60min']

            # Get ALL data (including indicators) for each interval and store in a list
            dfs = [get_all_data('AAPL', interval, window_size=60, month=month) for interval in intervals]

            # Combine all dataframes using outer join to align by datetime index
            combined_df = pd.concat(dfs, axis=1, join='outer')

            # Optionally, fill NaN values if required
            combined_df.fillna(method='ffill', inplace=True)
            combined_df.dropna(inplace=True)

            windowed_data = create_windows(combined_df, length=120)

            x, y = prepare_labels(windowed_data)

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            print(f'Epoch: {epoch}, Iteration: {iter}, Loss: {loss.item()}')

            if iter % 10 == 0:
                display_results(x, outputs)

            if month == '2023-07':
                #     save the weights
                torch.save(model.state_dict(), 'model_weights.pth')

            # Step the scheduler after each epoch

        scheduler.step()