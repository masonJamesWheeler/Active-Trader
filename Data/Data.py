import io
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from dateutil.relativedelta import relativedelta
from datetime import datetime
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from time import sleep
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


def get_stock_data(symbol, interval, month = '2023-07'):
    # Alpha Vantage Base URL
    stock_df, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize="full", month=month)
    # remove the numbers from the column names, i.e 1. open -> open
    stock_df = pd.DataFrame(stock_df)
    stock_df.columns = [col.split(' ')[1] for col in stock_df.columns]

    return stock_df

def get_all_months(start_year, start_month, end_year, end_month):
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    all_months = [(start + relativedelta(months=+x)).strftime('%Y-%m') for x in range((end.year-start.year)*12 + end.month - start.month + 1)]
    return all_months

def get_all_data(symbol, interval, window_size, month="2003-01"):
    df_stock = get_stock_data(symbol, interval, month=month)

    # Calculate SMA, EMA and RSI
    sma_window, ema_window = ti.get_sma(symbol=symbol, interval=interval, time_period=window_size, month=month)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
    sma_100, ema_100 = ti.get_sma(symbol=symbol, interval=interval, time_period=100, month=month)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=100, month=month)[0]
    sma_200, ema_200 = ti.get_sma(symbol=symbol, interval=interval, time_period=200, month=month)[0], \
    ti.get_ema(symbol=symbol, interval=interval, time_period=200, month=month)[0]
    vwap = ti.get_vwap(symbol=symbol, interval=interval, month=month)[0]
    rsi = ti.get_rsi(symbol=symbol, interval=interval, time_period=60, month=month)[0]
    macd = ti.get_macd(symbol=symbol, interval=interval, month=month)[0]
    bbands = ti.get_bbands(symbol=symbol, interval=interval, time_period=60, month=month)[0]
    wma = ti.get_wma(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
    cci = ti.get_cci(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
    aroon = ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
    obv = ti.get_obv(symbol=symbol, interval=interval, month=month)[0]
    stoch = ti.get_stoch(symbol=symbol, interval=interval, month=month)[0]
    stochf = ti.get_stochf(symbol=symbol, interval=interval, month=month)[0]
    stochrsi = ti.get_stochrsi(symbol=symbol, interval=interval, month=month)[0]
    df_stock['smawindow'], df_stock['emawindow'] = sma_window, ema_window
    df_stock['sma100'], df_stock['ema100'] = sma_100, ema_100
    df_stock['sma200'], df_stock['ema200'] = sma_200, ema_200
    df_stock['vwap'] = vwap
    df_stock['rsi'] = rsi
    df_stock['macd'], df_stock['macd_signal'], df_stock['macd_hist'] = macd["MACD"], macd["MACD_Signal"], macd[
        "MACD_Hist"]
    df_stock['bbands_upper'], df_stock['bbands_middle'], df_stock['bbands_lower'] = bbands['Real Upper Band'], bbands[
        'Real Middle Band'], bbands['Real Lower Band']
    df_stock['wma'] = wma
    df_stock['cci'] = cci
    df_stock['aroon_up'], df_stock['aroon_down'] = aroon['Aroon Up'], aroon['Aroon Down']
    df_stock['obv'] = obv
    df_stock['stoch_slowk'], df_stock['stoch_slowd'] = stoch['SlowK'], stoch['SlowD']
    df_stock['stochf_fastk'], df_stock['stochf_fastd'] = stochf['FastK'], stochf['FastD']
    df_stock['stochrsi_fastk'], df_stock['stochrsi_fastd'] = stochrsi['FastK'], stochrsi['FastD']

    # Convert the index to datetime if it's not already
    df_stock.index = pd.to_datetime(df_stock.index)

    # Define the start and end of regular trading hours (in hours)
    start_of_trading = 9.5  # 9:30 AM
    end_of_trading = 16  # 4:00 PM

    # Extract the hour from the index
    hour = df_stock.index.hour + df_stock.index.minute / 60.0

    # Filter the DataFrame to include only rows that fall within regular trading hours
    df_stock = df_stock[(hour >= start_of_trading) & (hour < end_of_trading)]

    # change the index to be numerical
    df_stock.reset_index(drop=True, inplace=True)

    # Drop the initial rows that have NaN values due to the rolling window calculations
    df_stock.dropna(inplace=True)

    # Get the Indices of the open, high, low, close, volume, and indicators columns
    columns_indices = {name: i for i, name in enumerate(df_stock.columns)}

    return df_stock, columns_indices


def get_and_process_data(ticker, interval, window_size, month):

    stock_data, columns = get_all_data(ticker, interval, window_size, month=month)
    # Drop the initial rows that have NaN values due to the rolling window calculations
    stock_data.dropna(inplace=True)
    stock_data = stock_data.to_numpy()

    if os.path.exists(f"Scalers/{ticker}_{interval}_scaler.pkl"):
        scaler = joblib.load(f"Scalers/{ticker}_{interval}_scaler.pkl")
        scaled_df = scaler.transform(stock_data)
        print("Loaded scaler")
    else:
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(stock_data)
        joblib.dump(scaler, f"Scalers/{ticker}_{interval}_scaler.pkl")
        print("Created scaler")

    # Convert the numpy array to a PyTorch tensor
    df = torch.from_numpy(stock_data)
    scaled_df = torch.from_numpy(scaled_df)

    return df, scaled_df, scaler

def get_last_data(symbol, interval, month='2023-07', window_size=128):
    '''
    Get the last windowed data from the given symbol and interval
    '''
    # Get the data from the given symbol and interval
    data, scaled_data, scaler = get_and_process_data(symbol, interval, window_size, month)
    # Get the last length amount of data
    last_data = data[-window_size:]
    last_scaled_data = scaled_data[-window_size:]
    return last_data, scaled_data, scaler

if __name__ == "__main__":
    AlphaVantage_Paid_Key = "A5QND05S0W7CU55E"
    ticker = "AAPL"
    interval = '1min'
    threshhold = 0.01
    window_size = 128
    years = 2
    months = 12
    month='1003-06'

    data = get_last_data(ticker, interval, month, window_size)
    print(data)
