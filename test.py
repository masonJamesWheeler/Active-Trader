import os
from time import sleep

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from Data.Data import get_and_process_data, get_all_data
from Data.Get_Fast_Data import get_most_recent_data, get_most_recent_data2
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

ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas')

api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets',
                         api_version='v2')

data = get_and_process_data("AAPL", "1min", 128, "2023-06")
