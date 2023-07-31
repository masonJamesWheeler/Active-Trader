import concurrent.futures
from datetime import datetime

import joblib
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

from PostGreSQL.Database import *

from datetime import datetime
import time

ticker = "aapl"
table_name = f'ticker_{ticker}_data'


def func1():
    ohlvc = get_latest_ohlcv(table_name)
    timestamp = ohlvc[0]
    open = ohlvc[1]
    high = ohlvc[2]
    low = ohlvc[3]
    close = ohlvc[4]
    volume = ohlvc[5]
    print(f'Latest OHLCV: {timestamp}, {open}, {high}, {low}, {close}, {volume}')


def func2():
    print("Func2 is running...")


def listener():
    current_minute = datetime.now().minute
    while True:
        if datetime.now().minute != current_minute:
            current_minute = datetime.now().minute
            print(f'Minute changed: {current_minute}')

            # Call your functions
            func1()
            func2()

        time.sleep(1)

if __name__ == '__main__':
    listener()
