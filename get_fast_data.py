import numpy as np
import pandas as pd
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
import time
import concurrent.futures
from functools import partial


ts = TimeSeries(key="A5QND05S0W7CU55E", output_format='pandas')
ti = TechIndicators(key='A5QND05S0W7CU55E', output_format='pandas')

def get_indicator_data(symbol, interval, window_size, ti_function, **kwargs):
    """
    Helper function to fetch indicator data.
    """
    if 'time_period' in kwargs:
        return ti_function(symbol=symbol, interval=interval, time_period=window_size, **kwargs)[0]
    else:
        return ti_function(symbol=symbol, interval=interval, **kwargs)[0]


def get_most_recent_data(symbol, interval, api_key, window_size):
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
        'sma200': 7,
        'ema200': 8,
        'sma800': 9,
        'ema800': 10,
        'vwap': 11,
        'rsi': 12,
        'macd': 13,
        'macd_signal': 14,
        'macd_hist': 15,
        'bbands_upper': 16,
        'bbands_middle': 17,
        'bbands_lower': 18,
        'adx': 19,
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
    indicator_to_function = {
        'smawindow': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=window_size)[0]['SMA'][-1],
        'emawindow': lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=window_size)[0]['EMA'][-1],
        'sma200': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=200)[0]['SMA'][-1],
        'ema200': lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=200)[0]['EMA'][-1],
        'sma800': lambda: ti.get_sma(symbol=symbol, interval=interval, time_period=800)[0]['SMA'][-1],
        'ema800' :lambda: ti.get_ema(symbol=symbol, interval=interval, time_period=800)[0]['EMA'][-1],
        'vwap' :lambda: ti.get_vwap(symbol=symbol, interval=interval)[0]["VWAP"][-1],
        'rsi' :lambda: ti.get_rsi(symbol=symbol, interval=interval)[0]["RSI"][-1],
        'macd' :lambda: ti.get_macd(symbol=symbol, interval=interval)[0]["MACD"][-1],
        'macd_signal' :lambda: ti.get_macd(symbol=symbol, interval=interval)[0]["MACD_Signal"][-1],
        'macd_hist' :lambda: ti.get_macd(symbol=symbol, interval=interval)[0]["MACD_Hist"][-1],
        'bbands_upper' :lambda: ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Upper Band'][-1],
        'bbands_middle' :lambda: ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Middle Band'][-1],
        'bbands_lower' :lambda: ti.get_bbands(symbol=symbol, interval=interval)[0]['Real Lower Band'][-1],
        'adx' :lambda: ti.get_adx(symbol=symbol, interval=interval, time_period=window_size)[0]['ADX'][-1],
        'cci' :lambda: ti.get_cci(symbol=symbol, interval=interval, time_period=window_size)[0]['CCI'][-1],
        'aroon_up' :lambda: ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size)[0]['Aroon Up'][-1],
        'aroon_down' :lambda: ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size)[0]['Aroon Down'][-1],
        'obv' :lambda: ti.get_obv(symbol=symbol, interval=interval)[0]['OBV'][-1],
        'stoch_slowk' :lambda: ti.get_stoch(symbol=symbol, interval=interval)[0]['SlowK'][-1],
        'stoch_slowd' :lambda: ti.get_stoch(symbol=symbol, interval=interval)[0]['SlowD'][-1],
        'stochf_fastk' :lambda: ti.get_stochf(symbol=symbol, interval=interval)[0]['FastK'][-1],
        'stochf_fastd' :lambda: ti.get_stochf(symbol=symbol, interval=interval)[0]['FastD'][-1],
        'stochrsi_fastk' :lambda: ti.get_stochrsi(symbol=symbol, interval=interval)[0]['FastK'][-1],
        'stochrsi_fastd' :lambda: ti.get_stochrsi(symbol=symbol, interval=interval)[0]['FastD'][-1]
    }

    def store_result(indicator, future):
        try:
            result = future.result()
        except Exception as exc:
            print(f'Generated an exception: {exc}')
        else:
            fast_data[indicator_to_index[indicator]] = result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for indicator, function in indicator_to_function.items():
            future = executor.submit(function)
            callback = partial(store_result, indicator)
            future.add_done_callback(callback)
            futures.append(future)
        concurrent.futures.wait(futures)

    return fast_data


if __name__ == "__main__":
    # get data
    df = get_most_recent_data(symbol="AAPL", interval="1min", api_key="A5QND05S0W7CU55E", window_size=14)
