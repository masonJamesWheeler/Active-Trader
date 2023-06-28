from time import sleep
import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')


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

# IMPLEMENTATION FOR TIME_SERIES_INTRADAY_EXTENDED
# def get_stock_data(symbol, interval, api_key, years=2, months=12):
    # # Empty DataFrame to hold all data
    # df_total = pd.DataFrame()
    #
    # for year in range(years, 0, -1):  # Two years = 2
    #     for month in range(months, 0, -1):  # Each year = 12 months
    #         slice = 'year' + str(year) + 'month' + str(month)
    #
    #         # Define API parameters
    #         params = {
    #             'function': 'TIME_SERIES_INTRADAY_EXTENDED',
    #             'symbol': symbol,
    #             'interval': interval,
    #             'slice': slice,
    #             'apikey': api_key,
    #             'adjusted': 'true',
    #             'datatype': 'csv'  # We want to receive the data in CSV format
    #         }
    #
    #         # Make the API request
    #         response = requests.get(base_url, params=params)
    #
    #         # Check if the request was successful
    #         if response.status_code == 200:
    #             df_slice = pd.read_csv(StringIO(response.text))
    #             df_slice['time'] = df_slice['time'].str.slice(0, -3)  # Remove seconds from timestamps
    #             df_total = pd.concat([df_total, df_slice])
    #         else:
    #             print(f'Request failed for slice: {slice}')
    #
    # # Reset index of the final dataframe
    # df_total.reset_index(drop=True, inplace=True)
    # return data

def get_stock_data(symbol, interval):
    # Alpha Vantage Base URL
    stock_df, metadata = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    # remove the numbers from the column names, i.e 1. open -> open
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = [col.split(' ')[1] for col in stock_df.columns]

    return stock_df


def get_all_data(symbol, interval, api_key, window_size):
    df_stock = get_stock_data(symbol, interval)

    # Calculate SMA, EMA and RSI
    df_stock['smawindow'] = df_stock['close'].rolling(window=window_size).mean()
    df_stock['emawindow'] = df_stock['close'].ewm(span=window_size, adjust=False).mean()
    df_stock['sma50'] = df_stock['close'].rolling(window=50).mean()
    df_stock['ema50'] = df_stock['close'].ewm(span=50, adjust=False).mean()
    df_stock['sma200'] = df_stock['close'].rolling(window=200).mean()
    df_stock['ema200'] = df_stock['close'].ewm(span=200, adjust=False).mean()
    df_stock['vwap'] = ti.get_vwap(symbol=symbol, interval=interval)[0]
    df_stock['rsi'] = ti.get_rsi(symbol=symbol, interval=interval)[0]
    df_stock['macd'] = ti.get_macd(symbol=symbol, interval=interval)[0]["MACD"]
    df_stock['macd_signal'] = ti.get_macd(symbol=symbol, interval=interval)[0]["MACD_Signal"]
    df_stock['macd_hist'] = ti.get_macd(symbol=symbol, interval=interval)[0]["MACD_Hist"]
    df_stock['bbands_upper'] = ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Upper Band']
    df_stock['bbands_middle'] = ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Middle Band']
    df_stock['bbands_lower'] = ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Lower Band']
    df_stock['adx'] = ti.get_adx(symbol=symbol, interval=interval, time_period=window_size)[0]

    # change the index to be numerical
    df_stock.reset_index(drop=True, inplace=True)

    # Drop the initial rows that have NaN values due to the rolling window calculations
    df_stock.dropna(inplace=True)

    # Reverse the order of the rows so because currently the most recent data is at the top
    df_stock = df_stock.iloc[::-1]


    # Get the Indices of the open, high, low, close, volume, and indicators columns
    columns_indices = {name: i for i, name in enumerate(df_stock.columns)}

    return df_stock, columns_indices


def get_and_process_data(tickers, interval, api_key, threshold, window_size, years=2, months=12):
    # create a list of shape, (num_windows, window_size, num_features)
    df_total = []

    for ticker in tickers:
        time = pd.Timestamp.now()
        df, columns_indices = get_all_data(ticker, interval, api_key, window_size)
        temp_df = create_windows(df, window_size)

        # Convert the data to a supported dtype if necessary
        if temp_df.dtype == np.float64:
            temp_df = temp_df.astype(np.float32)

        # wait for timer to hit 1 minute
        while pd.Timestamp.now() - time < pd.Timedelta(seconds=3):
            pass
    #   Add the windows for the current stock to the end of the list
        if ticker == tickers[0]:
            df_total = temp_df
        else:
            df_total = np.concatenate((df_total, temp_df), axis=0)

    #   Wait for 30 seconds before requesting data for the next stock
        while pd.Timestamp.now() - time < pd.Timedelta(seconds=30):
            pass

    return df_total

if __name__ == "__main__":
    AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
    tickers = ["AAPL"]
    interval = '1min'
    threshhold = 0.01
    window_size = 30
    years = 2
    months = 12

    stock_df = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)


