import numpy as np
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import concurrent.futures
from functools import partial
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas')

def get_indicator_data(symbol, interval, window_size, ti_function, **kwargs):
    """
    Helper function to fetch indicator data.
    """
    if 'time_period' in kwargs:
        return ti_function(symbol=symbol, interval=interval, time_period=window_size, **kwargs)[0]
    else:
        return ti_function(symbol=symbol, interval=interval, **kwargs)[0]


def get_most_recent_data(symbol, interval, api_key= "A5QND05S0W7CU55E", window_size=60):
    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')
    # Initialize an array of length 30 with NaN values
    fast_data = np.full((30,), np.nan)

    result, _ = ts.get_quote_endpoint(symbol=symbol)
    # extract the 2-6th values from the array
    df_stock_array = result.values[0][1:6].astype(np.float32)
    fast_data[:df_stock_array.size] = df_stock_array

    # Assume we have a mapping from indicator names to their index in the array
    indicator_to_index = {
        'smawindow': 5,
        'emawindow': 6,
        'sma100': 7,
        'ema100': 8,
        'sma200': 9,
        'ema200': 10,
        'vwap': 11,
        'rsi': 12,
        'macd': 13,
        'macd_signal': 14,
        'macd_hist': 15,
        'bbands_upper': 16,
        'bbands_middle': 17,
        'bbands_lower': 18,
        'wma': 19,
        'cci': 20,
        'aroon_up': 21,
        'aroon_down': 22,
        'obv': 23,
        'stoch_slowk': 24,
        'stoch_slowd': 25,
        'stochf_fastk': 26,
        'stochf_fastd': 27,
        'stochrsi_fastk': 28,
        'stochrsi_fastd': 29,
    }

    # dict that contains indicator names and the corresponding function calls
    indicator_to_function1 = {
        'smawindow': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=window_size)[0]['SMA'][-1],
        'emawindow': lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=window_size)[0]['EMA'][-1],
        'sma100': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=100)[0]['SMA'][-1],
        'ema100': lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=100)[0]['EMA'][-1],
        'sma200': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=200)[0]['SMA'][-1],
        'ema200' :lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=200)[0]['EMA'][-1],
        'vwap' :lambda: ti.get_vwap(symbol=symbol, interval=interval)[0]["VWAP"][-1],
        'rsi' :lambda: ti.get_rsi(symbol=symbol, time_period=60, interval=interval)[0]["RSI"][-1],
        'wma' :lambda: ti.get_wma(symbol=symbol, interval=interval, time_period = window_size)[0]['WMA'][-1],
        'cci' :lambda: ti.get_cci(symbol=symbol, interval=interval, time_period = window_size)[0]['CCI'][-1],
        'obv' :lambda: ti.get_obv(symbol=symbol, interval=interval)[0]['OBV'][-1],
    }

    # dict that contains indicator names and the corresponding function calls
    indicator_to_function2 = {
        'vwap': lambda: ti.get_vwap(symbol=symbol, interval=interval)[0]["VWAP"][-1],
        'rsi': lambda: ti.get_rsi(symbol=symbol, time_period = 60, interval=interval)[0]["RSI"][-1],
        'macd': lambda: ti.get_macd(symbol=symbol, interval=interval)[0],
        'bbands': lambda: ti.get_bbands(symbol=symbol, interval=interval, time_period=60)[0],
        'aroon': lambda: ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size)[0],
        'stoch': lambda: ti.get_stoch(symbol=symbol, interval=interval)[0],
        'stochf': lambda: ti.get_stochf(symbol=symbol, interval=interval)[0],
        'stochrsi': lambda: ti.get_stochrsi(symbol=symbol, interval=interval)[0],
    }

    indicator_key = {
        'macd': {
            'macd': 'MACD',
            'macd_signal': 'MACD_Signal',
            'macd_hist': 'MACD_Hist'
        },
        'bbands': {
            'bbands_upper': 'Real Upper Band',
            'bbands_middle': 'Real Middle Band',
            'bbands_lower': 'Real Lower Band',
        },
        'aroon': {
            'aroon_up': 'Aroon Up',
            'aroon_down': 'Aroon Down',
        },
        'stoch': {
            'stoch_slowk': 'SlowK',
            'stoch_slowd': 'SlowD',
        },
        'stochf': {
            'stochf_fastk': 'FastK',
            'stochf_fastd': 'FastD',
        },
        'stochrsi': {
            'stochrsi_fastk': 'FastK',
            'stochrsi_fastd': 'FastD',
        },
    }

    def call_function_and_store_results(function, keys):
        data = function()
        for indicator, key in keys.items():
            result = data[key][-1]
            fast_data[indicator_to_index[indicator]] = result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(call_function_and_store_results, indicator_to_function2[func_name], indicator_key[func_name])
            for func_name in ['macd', 'bbands', 'aroon', 'stoch', 'stochf', 'stochrsi']
        ]
        concurrent.futures.wait(futures)

    def store_result(indicator, future):
        try:
            result = future.result()
        except Exception as exc:
            print(f'Generated an exception: {exc}')
        else:
            fast_data[indicator_to_index[indicator]] = result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for indicator, function in indicator_to_function1.items():
            future = executor.submit(function)
            callback = partial(store_result, indicator)
            future.add_done_callback(callback)
            futures.append(future)
        concurrent.futures.wait(futures)

    return np.array(fast_data)

if __name__ == "__main__":
    print(get_most_recent_data("AAPL", "1min"))