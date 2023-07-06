import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from time import sleep
from datetime import datetime
from dotenv import load_dotenv
import os

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
    stock_df, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
    # remove the numbers from the column names, i.e 1. open -> open
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = [col.split(' ')[1] for col in stock_df.columns]

    return stock_df


def get_all_data(symbol, interval, api_key, window_size):
    df_stock = get_stock_data(symbol, interval)

    # Calculate SMA, EMA and RSI
    sma_window, ema_window = ti.get_sma(symbol=symbol, interval=interval, time_period=window_size)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=window_size)[0]
    sma_200, ema_200 = ti.get_sma(symbol=symbol, interval=interval, time_period=200)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=200)[0]
    sma_800, ema_800 = ti.get_sma(symbol=symbol, interval=interval, time_period=800)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=800)[0]
    vwap = ti.get_vwap(symbol=symbol, interval=interval)[0]
    rsi = ti.get_rsi(symbol=symbol, interval=interval, time_period=60)[0]
    macd = ti.get_macd(symbol=symbol, interval=interval)[0]
    bbands = ti.get_bbands(symbol=symbol, interval=interval, time_period=60)[0]
    adx = ti.get_adx(symbol=symbol, interval=interval, time_period=window_size)[0]
    cci = ti.get_cci(symbol=symbol, interval=interval, time_period=window_size)[0]
    aroon = ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size)[0]
    obv = ti.get_obv(symbol=symbol, interval=interval)[0]
    stoch = ti.get_stoch(symbol=symbol, interval=interval)[0]
    stochf = ti.get_stochf(symbol=symbol, interval=interval)[0]
    stochrsi = ti.get_stochrsi(symbol=symbol, interval=interval)[0]

    df_stock['smawindow'], df_stock['emawindow'] = sma_window, ema_window
    df_stock['sma200'], df_stock['ema200'] = sma_200, ema_200
    df_stock['sma800'], df_stock['ema800'] = sma_800, ema_800
    df_stock['vwap'] = vwap
    df_stock['rsi'] = rsi
    df_stock['macd'], df_stock['macd_signal'], df_stock['macd_hist'] = macd["MACD"], macd["MACD_Signal"], macd[
        "MACD_Hist"]
    df_stock['bbands_upper'], df_stock['bbands_middle'], df_stock['bbands_lower'] = bbands['Real Upper Band'], bbands[
        'Real Middle Band'], bbands['Real Lower Band']
    df_stock['adx'] = adx
    df_stock['cci'] = cci
    df_stock['aroon_up'], df_stock['aroon_down'] = aroon['Aroon Up'], aroon['Aroon Down']
    df_stock['obv'] = obv
    df_stock['stoch_slowk'], df_stock['stoch_slowd'] = stoch['SlowK'], stoch['SlowD']
    df_stock['stochf_fastk'], df_stock['stochf_fastd'] = stochf['FastK'], stochf['FastD']
    df_stock['stochrsi_fastk'], df_stock['stochrsi_fastd'] = stochrsi['FastK'], stochrsi['FastD']

    # change the index to be numerical
    df_stock.reset_index(drop=True, inplace=True)

    # Drop the initial rows that have NaN values due to the rolling window calculations
    df_stock.dropna(inplace=True)

    # Reverse the order of the rows so because currently the most recent data is at the top
    df_stock = df_stock.iloc[::-1]


    # Get the Indices of the open, high, low, close, volume, and indicators columns
    columns_indices = {name: i for i, name in enumerate(df_stock.columns)}

    return df_stock, columns_indices


def get_and_process_data(ticker, interval, api_key, threshold, window_size, years=2, months=12):
    # create a list of shape, (num_windows, window_size, num_features)
    df = []
    scaled_df = []
    time = pd.Timestamp.now()
    df, columns_indices = get_all_data(ticker, interval, api_key, window_size)

    temp_df = create_windows(df, window_size)

    # Flatten the window_size and num_features dimensions into one
    num_windows, _, _ = temp_df.shape
    temp_df_2d = temp_df.reshape(num_windows, -1)
    # Scale the DataFrame
    scaler = MinMaxScaler()
    scaled_temp_df_2d = scaler.fit_transform(temp_df_2d)

    # Reshape the scaled data back to the original shape
    scaled_temp_df = scaled_temp_df_2d.reshape(temp_df.shape)

    # Convert the data to a supported dtype if necessary
    if temp_df.dtype == np.float64:
        df = temp_df.astype(np.float32)
        scaled_df = scaled_temp_df.astype(np.float32)

    # Convert the numpy array to a PyTorch tensor
    df = torch.from_numpy(temp_df)
    scaled_df = torch.from_numpy(scaled_temp_df)

    return df, scaled_df, scaler

if __name__ == "__main__":
    AlphaVantage_Paid_Key = "A5QND05S0W7CU55E"
    tickers = ["UBER"]
    interval = '1min'
    threshhold = 0.01
    window_size = 30
    years = 2
    months = 12

    # data = get_and_process_data(tickers[0], interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)

    vwap = (ti.get_vwap(symbol=tickers[0], interval=interval))


