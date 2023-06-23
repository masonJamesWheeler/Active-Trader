from time import sleep
import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.utils import to_categorical
from datetime import datetime


# Alpha Vantage Base URL
base_url = 'https://www.alphavantage.co/query?'

def convert_time_to_trading_minutes(time_str):
    # Try to extract the hour and minute from the string, allowing for times that don't include seconds
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

    hour, minute = dt.hour, dt.minute

    # Convert the time to the minute of the trading day, subtracting the starting minute of the trading day
    minute_of_day = (hour - 13) * 60 + minute - 30

    return minute_of_day

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


def get_stock_data(symbol, interval, api_key, years=2, months=12):
    # Empty DataFrame to hold all data
    df_total = pd.DataFrame()

    for year in range(years, 0, -1):  # Two years = 2
        for month in range(months, 0, -1):  # Each year = 12 months
            slice = 'year' + str(year) + 'month' + str(month)

            # Define API parameters
            params = {
                'function': 'TIME_SERIES_INTRADAY_EXTENDED',
                'symbol': symbol,
                'interval': interval,
                'slice': slice,
                'apikey': api_key,
                'adjusted': 'true',
                'datatype': 'csv'  # We want to receive the data in CSV format
            }

            # Make the API request
            response = requests.get(base_url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                df_slice = pd.read_csv(StringIO(response.text))
                df_slice['time'] = df_slice['time'].str.slice(0, -3)  # Remove seconds from timestamps
                df_total = pd.concat([df_total, df_slice])
            else:
                print(f'Request failed for slice: {slice}')

    # Reset index of the final dataframe
    df_total.reset_index(drop=True, inplace=True)

    return df_total


def get_all_data(symbol, interval, api_key, window_size):
    df_stock = get_stock_data(symbol, interval, api_key)

    # Calculate SMA, EMA and RSI
    df_stock['smawindow'] = df_stock['close'].rolling(window=window_size).mean()
    df_stock['emawindow'] = df_stock['close'].ewm(span=window_size, adjust=False).mean()
    df_stock['sma50'] = df_stock['close'].rolling(window=50).mean()
    df_stock['ema50'] = df_stock['close'].ewm(span=50, adjust=False).mean()
    df_stock['sma200'] = df_stock['close'].rolling(window=200).mean()
    df_stock['ema200'] = df_stock['close'].ewm(span=200, adjust=False).mean()

    delta = df_stock['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.rolling(window=window_size).mean()
    avg_loss = abs(down.rolling(window=window_size).mean())
    avg_twice_gain = up.rolling(window=window_size * 2).mean()
    avg_twice_loss = abs(down.rolling(window=window_size * 2).mean())
    rs = avg_gain / avg_loss
    rs2 = avg_twice_gain / avg_twice_loss

    df_stock['rsi'] = 100 - (100 / (1 + rs))
    df_stock['rsi2'] = 100 - (100 / (1 + rs2))

    df_stock['time'] = df_stock['time'].apply(convert_time_to_trading_minutes)

    # Drop the initial rows that have NaN values due to the rolling window calculations
    df_stock.dropna(inplace=True)

    # scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df_stock = pd.DataFrame(scaler.fit_transform(df_stock), columns=df_stock.columns, index=df_stock.index)

    # Get the Indices of the open, high, low, close, volume, and indicators columns
    columns_indices = {name: i for i, name in enumerate(df_stock.columns)}

    return df_stock, columns_indices


def get_and_process_data(tickers, interval, api_key, threshold, window_size, years=2, months=12):
    column_indices = {}

    # Define lists to hold training and testing data
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for ticker in tickers:
        # start clock
        time = pd.Timestamp.now()
        df, columns_indices = get_all_data(ticker, interval, api_key, window_size)

        temp_windows = create_windows(df, window_size)
        target = np.zeros(len(df) - window_size)

        for i in range(len(temp_windows) - window_size):
            #     Check if the average price of the window an hour in the future is greater than the current price
            if np.mean(temp_windows[i + window_size, :, columns_indices['close']]) > temp_windows[
                i, -1, columns_indices['close']] * (
                    1 + threshold):
                target[i] = 1
            elif np.mean(temp_windows[i + window_size, :, columns_indices['close']]) < temp_windows[
                i, -1, columns_indices['close']] * (
                    1 - threshold):
                target[i] = 2
            else:
                target[i] = 0

        # Convert the data to a supported dtype if necessary
        if temp_windows.dtype == np.float64:
            temp_windows = temp_windows.astype(np.float32)

        if isinstance(target, np.ndarray) and target.dtype == np.float64:
            target = target.astype(np.float32)

        # Split the data into training and testing set with a 80/20 ratio
        train_size = int(len(temp_windows) * 0.8)
        X_train.extend(temp_windows[:train_size])
        Y_train.extend(target[:train_size])
        X_test.extend(temp_windows[train_size:])
        Y_test.extend(target[train_size:])

        # wait for timer to hit 1 minute
        while pd.Timestamp.now() - time < pd.Timedelta(minutes=1):
            pass

        print(ticker + " done")

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    Y_train = to_categorical(Y_train, num_classes=3)
    Y_test = to_categorical(Y_test, num_classes=3)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
    tickers = ["AAPL", "AMZN", "GOOG", "MSFT", "NFLX", "NVDA", "TSLA"]
    interval = '1min'
    threshhold = 0.003
    window_size = 30
    years = 2
    months = 12

    X_train, Y_train, X_test, Y_test = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
