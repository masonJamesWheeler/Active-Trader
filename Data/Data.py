from datetime import datetime

import joblib
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from Utilities.Indicators import *

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
    stock_df = stock_df.iloc[::-1]

    return stock_df

def get_all_months(start_year, start_month, end_year, end_month):
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    all_months = [(start + relativedelta(months=+x)).strftime('%Y-%m') for x in range((end.year-start.year)*12 + end.month - start.month + 1)]
    return all_months

def get_all_data(symbol, interval, window_size, month="2003-01"):
    # Assuming that get_stock_data is a function that gets the OHLCV data
    data = get_stock_data(symbol, interval, month=month)

    # Replace the API calls with the function calls
    data['smawindow'] = SMA(data, window_size)
    data['emawindow'] = EMA(data, window_size)
    data['sma100'] = SMA(data, 100)
    data['ema100'] = EMA(data, 100)
    data['sma200'] = SMA(data, 200)
    data['ema200'] = EMA(data, 200)
    data['vwap'] = VWAP(data)
    data['rsi'] = RSI(data, 60)
    macd_line, signal_line, histogram = MACD(data)
    data['macd'] = macd_line
    data['macd_signal'] = signal_line
    data['macd_hist'] = histogram
    upper_band, middle_band, lower_band = Bollinger_Bands(data, 60)
    data['bbands_upper'] = upper_band
    data['bbands_middle'] = middle_band
    data['bbands_lower'] = lower_band
    data['wma'] = WMA(data, window_size)
    data['cci'] = CCI(data, window_size)
    aroon_up, aroon_down = Aroon(data, window_size)
    data['aroon_up'] = aroon_up
    data['aroon_down'] = aroon_down
    data['obv'] = OBV(data)
    fastk, fastd = Stochastic(data)
    data['stoch_slowk'] = fastk
    data['stoch_slowd'] = fastd
    fastk, fastd = Fast_Stochastic(data)
    data['stochf_fastk'] = fastk
    data['stochf_fastd'] = fastd
    fastk, fastd = Stochastic_RSI(data, 60)
    data['stochrsi_fastk'] = fastk
    data['stochrsi_fastd'] = fastd

    # Convert the index to datetime if it's not already
    data.index = pd.to_datetime(data.index)

    # Define the start and end of regular trading hours (in hours)
    start_of_trading = 9.5  # 9:30 AM
    end_of_trading = 16  # 4:00 PM

    # Extract the hour from the index
    hour = data.index.hour + data.index.minute / 60.0

    # Filter the DataFrame to include only rows that fall within regular trading hours
    data = data[(hour >= start_of_trading) & (hour < end_of_trading)]

    # change the index to be numerical
    data.reset_index(drop=True, inplace=True)

    # Drop the initial rows that have NaN values due to the rolling window calculations
    data.dropna(inplace=True)

    # Get the Indices of the open, high, low, close, volume, and indicators columns
    columns_indices = {name: i for i, name in enumerate(data.columns)}

    return data, columns_indices


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
    df = torch.from_numpy(stock_data).float()
    scaled_df = torch.from_numpy(scaled_df).float()

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
    alpha_vantage_api_key = alpha_vantage_api_key
