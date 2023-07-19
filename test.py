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
import matplotlib.pyplot as plt



# Load environment variables from .env file
load_dotenv()
import requests
# Access the API keys from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
paper_alpaca_key = os.getenv("PAPER_ALPACA_KEY")
paper_alpaca_secret_key = os.getenv("PAPER_ALPACA_SECRET_KEY")

base_url = 'https://paper-api.alpaca.markets'

ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas')

api = tradeapi.REST(paper_alpaca_key, paper_alpaca_secret_key, base_url='https://paper-api.alpaca.markets',
                         api_version='v2')
print(api.get_account())

# # Define the data
# years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
# publications = [1, 23, 76, 195, 391, 488, 638, 775, 895, 926]
# citations = [1,75, 1056, 2588, 5319, 8839, 13998, 22649, 32156, 36122]
#
# # Create a figure and a set of subplots
# fig, ax1 = plt.subplots(figsize=[10,16], dpi=300)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['lines.linewidth'] = 8
#
# # Plot publications
# color = 'tab:blue'
# ax1.set_xlabel('Years', fontsize=24)
# ax1.set_ylabel('Publications', color=color, fontsize=22)
# ax1.plot(years, publications, color=color)
# ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
# ax1.tick_params(axis='x', labelsize=20)
#
# # Instantiate a second axes that shares the same x-axis
# ax2 = ax1.twinx()
# color = 'tab:red'
#
# # Plot citations
# ax2.set_ylabel('Citations', color=color, fontsize=22)
# ax2.plot(years, citations, color=color)
# ax2.tick_params(axis='y', labelcolor=color, labelsize=20)
#
# # Leave padding for the axis labels
# fig.tight_layout(pad=4.0)
#
# plt.title('Crispr Cas 9 Publications and Citations Per Year', fontsize=22)
# plt.savefig('Crispr Cas 9 Publications and Citations Per Year.png', dpi=300)



