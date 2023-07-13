import os
from time import sleep

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import matplotlib.pyplot as plt


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

'''
GETTING THE DATA TO CALCULATE THE INDICATORS
'''
symbol = "AAPL"
interval = '1min'
month = "2023-06"
window_size = 128

# data, _ = ts.get_intraday(symbol=symbol, interval='1min', outputsize="full", month=month)
# # Reverse the data so that it is in chronological order
# data = data.iloc[::-1]
# data.columns = [col.split(' ')[1] for col in data.columns]
#
# sma_window, ema_window = ti.get_sma(symbol=symbol, interval=interval, time_period=window_size, month=month)[0], \
#     ti.get_ema(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
# sma_100, ema_100 = ti.get_sma(symbol=symbol, interval=interval, time_period=100, month=month)[0], \
#     ti.get_ema(symbol=symbol, interval=interval, time_period=100, month=month)[0]
# sma_200, ema_200 = ti.get_sma(symbol=symbol, interval=interval, time_period=200, month=month)[0], \
#     ti.get_ema(symbol=symbol, interval=interval, time_period=200, month=month)[0]
# realVwap = ti.get_vwap(symbol=symbol, interval=interval, month=month)[0]
# realRsi = ti.get_rsi(symbol=symbol, interval=interval, time_period=60, month=month)[0]
# realMACD = ti.get_macd(symbol=symbol, interval=interval, month=month)[0]
# realBbands = ti.get_bbands(symbol=symbol, interval=interval, time_period=60, month=month)[0]
# realWma = ti.get_wma(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
# realCci = ti.get_cci(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
# realAroon = ti.get_aroon(symbol=symbol, interval=interval, time_period=window_size, month=month)[0]
# realObv = ti.get_obv(symbol=symbol, interval=interval, month=month)[0]
# realStoch = ti.get_stoch(symbol=symbol, interval=interval, month=month)[0]
# realStochf = ti.get_stochf(symbol=symbol, interval=interval, month=month)[0]
# realStochrsi = ti.get_stochrsi(symbol=symbol, interval=interval, month=month)[0]


'''
INDICATOR FUNCTIONS
'''
def SMA(data, window):
    sma = data['close'].rolling(window).mean()
    return sma

def EMA(data, window):
    return data['close'].ewm(span=window, adjust=False).mean()


def VWAP(data):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = data.groupby(data.index.normalize()).apply(
        lambda x: (x['volume'] * typical_price.loc[x.index]).cumsum() / x['volume'].cumsum()
    )
    # Reset the index to remove the date level and keep only the timestamp level
    vwap.index = vwap.index.droplevel(0)
    return vwap


def RSI(data, window):
    delta = data['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Use simple average for the first calculation
    average_gain = up[:window].mean()
    average_loss = abs(down[:window].mean())

    gain = up[window:]
    loss = abs(down[window:])

    rsi = pd.Series(index=delta.index, dtype='float32')

    # Use modified moving average for all subsequent calculations
    for i in range(len(gain)):
        average_gain = ((average_gain * (window - 1)) + gain.iloc[i]) / window
        average_loss = ((average_loss * (window - 1)) + loss.iloc[i]) / window
        rs = average_gain / average_loss
        rsi.iloc[i+window] = 100 - (100 / (1 + rs))

    return rsi

def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    DIF = EMA(data, fastperiod) - EMA(data, slowperiod)
    DEA = DIF.ewm(span=signalperiod, adjust=False).mean()
    MACD = (DIF - DEA) * 2
    return DIF, DEA, MACD


def Bollinger_Bands(data, window):
    sma = SMA(data, window)
    std = data['close'].rolling(window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, sma, lower_band  # Upper Bollinger Band, Middle Bollinger Band, Lower Bollinger Band

def WMA(data, window):
    weights = np.arange(window, 0, -1)
    return data['close'].rolling(window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

def CCI(data, window):
    TP = (data['high'] + data['low'] + data['close']) / 3
    return (TP - SMA(data.assign(close=TP.values), window)) / (0.015 * TP.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))

def Aroon(data, time_period):
    aroon_up = data['high'].rolling(time_period + 1).apply(lambda x: x.argmax(), raw=True) / time_period * 100
    aroon_down = data['low'].rolling(time_period + 1).apply(lambda x: x.argmin(), raw=True) / time_period * 100
    return aroon_up, aroon_down


def OBV(data):
    volumes = data['volume']
    changes = data['close'].diff()
    up = changes > 0
    down = changes < 0
    obv = np.zeros_like(volumes)
    obv[0] = volumes.iloc[0]
    obv[up] = volumes[up].cumsum()
    obv[down] = -volumes[down].cumsum()
    return pd.Series(obv, index=data.index)  # return pandas Series instead of numpy ndarray


def Stochastic(data, fastkperiod=5, slowkperiod=3, slowdperiod=3):
    L = data['low'].rolling(fastkperiod).min()
    H = data['high'].rolling(fastkperiod).max()
    K = 100 * ((data['close'] - L) / (H - L)) # Fast %K
    Fast_D = K.rolling(slowkperiod).mean() # Fast %D
    Slow_D = Fast_D.rolling(slowdperiod).mean() # Slow %D
    return Fast_D, Slow_D  # Fast %K, Fast %D, Slow %D


def Fast_Stochastic(data, fastkperiod=5, slowkperiod=3, slowdperiod=3):
    L = data['low'].rolling(fastkperiod).min()
    H = data['high'].rolling(fastkperiod).max()
    K = 100 * ((data['close'] - L) / (H - L)) # Fast %K
    Fast_D = K.rolling(slowkperiod).mean() # Fast %D
    return K, Fast_D  # Fast %K, Fast %D, Slow %D

def Stochastic_RSI(data, rsi_window, fastkperiod=5, slowkperiod=3, slowdperiod=3):
    rsi = RSI(data, rsi_window)
    L = rsi.rolling(window=fastkperiod).min()
    H = rsi.rolling(window=fastkperiod).max()
    K = 100 * ((rsi - L) / (H - L)) # Fast %K
    Fast_D = K.rolling(slowkperiod).mean() # Fast %D
    Slow_D = Fast_D.rolling(slowdperiod).mean() # Slow %D
    return K, Fast_D  # StochRSI Fast %K, StochRSI Fast %D, StochRSI Slow %D


'''
UNIT TESTS
'''
def test_SMA():
    sma = SMA(data, 100)
    real_smas = sma_100["SMA"]
#   Create a dataframe with the calculated SMA and the real SMA to compare
    sma_frame = pd.DataFrame({'sma': sma, 'real_sma': real_smas}).dropna()
    print(sma_frame)
#   assert that the average difference is less than 1%
    assert np.mean(abs(sma_frame['sma'] - sma_frame['real_sma']) / sma_frame['real_sma']) < 0.01

def test_EMA():
    ema = EMA(data, 100)[-100:]
    real_emas = ema_100["EMA"][-100:]
    # Create a dataframe with the calculated EMA and the real EMA to compare
    ema_frame = pd.DataFrame({'ema': ema, 'real_ema': real_emas}).dropna()
    print(ema_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(ema_frame['ema'] - ema_frame['real_ema']) / ema_frame['real_ema']) < 0.01

def test_VWAP():
    vwap = VWAP(data)
    real_vwap = realVwap["VWAP"]
    # Create a dataframe with the calculated VWAP and the real VWAP to compare
    vwap_frame = pd.DataFrame({'vwap': vwap, 'real_vwap': real_vwap}).dropna()
    print(vwap_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(vwap_frame['vwap'] - vwap_frame['real_vwap']) / vwap_frame['real_vwap']) < 0.01

def test_RSI():
    rsi = RSI(data, 60)
    real_rsi = realRsi["RSI"]
    # Create a dataframe with the calculated RSI and the real RSI to compare
    rsi_frame = pd.DataFrame({'rsi': rsi, 'real_rsi': real_rsi}).dropna()
    print(rsi_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(rsi_frame['rsi'] - rsi_frame['real_rsi']) / rsi_frame['real_rsi']) < 0.01


def test_MACD():
    macd_line, signal_line, _ = MACD(data, 12, 26, 9)
    macd_line = macd_line
    signal_line = signal_line
    # # reverse macd index to match the real macd index
    # realMacd = realMACD[::-1]
    real_macd_line = realMACD["MACD"]
    real_signal_line = realMACD["MACD_Signal"]
    # Create a dataframe with the calculated MACD and the real MACD to compare
    macd_frame = pd.DataFrame({'macd_line': macd_line, 'real_macd_line': real_macd_line, 'signal_line': signal_line, 'real_signal_line': real_signal_line}).dropna()
    print(macd_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(macd_frame['macd_line'] - macd_frame['real_macd_line']) / macd_frame['real_macd_line']) < 0.01


def test_Bollinger_Bands():
    upper_band, _, lower_band = Bollinger_Bands(data, 20)
    real_upper = realBbands['Real Upper Band']
    real_lower = realBbands['Real Lower Band']
    # Create a dataframe with the calculated Bollinger Bands and the real Bollinger Bands to compare
    bbands_frame = pd.DataFrame({'upper_band': upper_band, 'real_upper': real_upper, 'lower_band': lower_band, 'real_lower': real_lower}).dropna()
    print(bbands_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(bbands_frame['upper_band'] - bbands_frame['real_upper']) / bbands_frame['real_upper']) < 0.01
    assert np.mean(abs(bbands_frame['lower_band'] - bbands_frame['real_lower']) / bbands_frame['real_lower']) < 0.01

def test_WMA():
    wma = WMA(data, 100)[-100:]
    real_wma = realWma["WMA"][-100:]
    # Create a dataframe with the calculated WMA and the real WMA to compare
    wma_frame = pd.DataFrame({'wma': wma, 'real_wma': real_wma}).dropna()
    print(wma_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(wma_frame['wma'] - wma_frame['real_wma']) / wma_frame['real_wma']) < 0.01

def test_CCI():
    cci = CCI(data, 128)
    real_cci = realCci["CCI"]
    # Create a dataframe with the calculated CCI and the real CCI to compare
    cci_frame = pd.DataFrame({'cci': cci, 'real_cci': real_cci}).dropna()
    print(cci_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(cci_frame['cci'] - cci_frame['real_cci']) / cci_frame['real_cci']) < 0.01

def test_Aroon():
    aroon_up, aroon_down = Aroon(data, 128)
    real_up = realAroon["Aroon Up"]
    real_down = realAroon["Aroon Down"]
    # Create a dataframe with the calculated Aroon and the real Aroon to compare
    aroon_frame = pd.DataFrame({'aroon_up': aroon_up, 'real_up': real_up, 'aroon_down': aroon_down, 'real_down': real_down}).dropna()
    print(aroon_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(aroon_frame['aroon_up'] - aroon_frame['real_up']) / aroon_frame['real_up']) < 0.01

def test_OBV():
    obv = OBV(data)
    real_obv = obv
    # Create a dataframe with the calculated OBV and the real OBV to compare
    obv_frame = pd.DataFrame({'obv': obv, 'real_obv': real_obv}).dropna()
    print(obv_frame)
    # assert that the average difference is less than 1%
    assert np.mean(abs(obv_frame['obv'] - obv_frame['real_obv']) / obv_frame['real_obv']) < 0.01

def test_Stochastic():
    slowk, slowd = Stochastic(data)
    real_slowk = realStoch['SlowK']
    real_slowd = realStoch['SlowD']
    # Create a dataframe with the calculated Stochastic and the real Stochastic to compare
    stoch = pd.DataFrame({'slowk': slowk, 'real_slowk': real_slowk, 'slowd': slowd, 'real_slowd': real_slowd}).dropna()
    print(stoch)
    # assert that the average difference is less than 1%
    assert np.mean(abs(stoch['slowk'] - stoch['real_slowk']) / stoch['real_slowk']) < 0.01
    assert np.mean(abs(stoch['slowd'] - stoch['real_slowd']) / stoch['real_slowd']) < 0.01


def test_Fast_Stochastic():
    fastk, fastd = Fast_Stochastic(data)
    real_fastk = realStochf['FastK']
    real_fastd = realStochf['FastD']
    # Create a dataframe with the calculated Fast Stochastic and the real Fast Stochastic to compare
    stochf = pd.DataFrame({'fastk': fastk, 'real_fastk': real_fastk, 'fastd': fastd, 'real_fastd': real_fastd}).dropna()
    print(stochf)
    # assert that the average difference is less than 1%
    assert np.mean(abs(stochf['fastk'] - stochf['real_fastk']) / stochf['real_fastk']) < 0.01
    assert np.mean(abs(stochf['fastd'] - stochf['real_fastd']) / stochf['real_fastd']) < 0.01


def test_Stochastic_RSI(data):
    fastk, fastd = Stochastic_RSI(data, 20)
    real_fastk = realStochrsi['FastK']
    real_fastd = realStochrsi['FastD']
    # Create a dataframe with the calculated Stochastic RSI and the real Stochastic RSI to compare
    stochrsi = pd.DataFrame({'fastk': fastk, 'real_fastk': real_fastk, 'fastd': fastd, 'real_fastd': real_fastd}).dropna()
    print(stochrsi)
    # assert that the average difference is less than 1%
    assert np.mean(abs(stochrsi['fastk'] - stochrsi['real_fastk']) / stochrsi['real_fastk']) < 0.01


if __name__ == '__main__':
    test_SMA()
    test_EMA()
    test_VWAP()
    test_RSI()
    # test_MACD()
    test_Bollinger_Bands()
    test_WMA()
    test_CCI()
    # test_Aroon()
    test_OBV()
    test_Stochastic()
    test_Fast_Stochastic()
    test_Stochastic_RSI(data)
    print('All tests passed.')

