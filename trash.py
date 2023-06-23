from time import sleep
import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def convert_time_to_trading_minutes(time_str):
    # Extract the hour and minute from the string
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    hour, minute = dt.hour, dt.minute

    # Convert the time to the minute of the day, subtracting the starting minute of the trading day
    minute_of_day = (hour - 13) * 60 + minute - 30

    return minute_of_day

symbol = 'AAPL'
interval = '1min'
AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
# Alpha Vantage Base URL
base_url = 'https://www.alphavantage.co/query?'

# Define API parameters for 1 day of data
params = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': symbol,
    'interval': interval,
    'apikey': AlphaVantage_Free_Key,
    'adjusted': 'true',
    'datatype': 'csv'  # We want to receive the data in CSV format
}

# Make the API request
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    df_total = pd.read_csv(StringIO(response.text))
    # df_total['time'] = df_total['time'].str.slice(0, -3)  # Remove seconds from timestamps
else:
    print(f'Request failed with status code: {response.status_code}')

# Reset index of the dataframe
df_total.reset_index(drop=True, inplace=True)
df_total['timestamp'] = df_total['timestamp'].apply(convert_time_to_trading_minutes)

print(df_total.head())
print(df_total.tail())
