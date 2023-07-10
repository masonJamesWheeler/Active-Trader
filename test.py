import os
from time import sleep

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from Data.Data import get_and_process_data, get_all_data
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')
api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets',
                         api_version='v2')

def get_last_data(symbol, interval, month='2023-07', length=128):
    '''
    Get the last windowed data from the given symbol and interval
    '''
    # Get the data from the given symbol and interval
    data = get_and_process_data(symbol, interval, month)
    # Get the last length amount of data
    last_data = data[-length:]
    return last_data

# latest_quote = ts.get_intraday(symbol='AAPL', interval='1min', outputsize='full')[0].iloc[0]
# print(latest_quote)


