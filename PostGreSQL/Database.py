from datetime import datetime
from time import sleep

import psycopg2
import torch

# connection parameters
params = {
    "database": "tickers",
    "user": "wheels",
    "password": "",  # Provide password if any, else keep it as empty string
    "host": "localhost",  # or the IP of your DB
    "port": "5432"  # default postgres port
}

# create a connection
conn = psycopg2.connect(**params)


def add_row_to_table(table_name, values, action, shares, max_rows=50000):
    cursor = conn.cursor()

    # Check the number of rows
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]

    # If the row count exceeds the limit, delete the oldest row
    if row_count >= max_rows:
        cursor.execute(f"DELETE FROM {table_name} WHERE timestamp = (SELECT MIN(timestamp) FROM {table_name})")

    # Convert tensor to numpy if needed
    if isinstance(values, torch.Tensor):
        values = values.numpy()

    if isinstance(action, torch.Tensor):
        action = action.item()

    if isinstance(shares, torch.Tensor):
        shares = shares.item()

    # Prepare the values
    values_str = ', '.join(map(str, values.tolist()))
    values_str += f', {action}, {shares}'

    # Generate the timestamp
    timestamp = datetime.now()

    # Insert the new row
    try:
        cursor.execute(f"""
            INSERT INTO {table_name} (
                timestamp, open_value, high_value, low_value, close_value, 
                volume, sma_window, ema_window, sma_100, ema_100, sma_200, 
                ema_200, vwap, rsi_60, wma_window, cci_128, obv, macd_line, 
                signal_line, histogram, upper_band, middle_band, lower_band, 
                aroon_up, aroon_down, stoch_fastk, stoch_fastd, fast_stochk, 
                fast_stochd, stoch_rsi_fastk, stoch_rsi_fastd, minutes_since_start, 
                hour_sin, hour_cos, minute_sin, minute_cos, action, shares
            ) VALUES (
                '{timestamp}', {values_str}
            )
        """)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Values string was: {values_str}")

    # Commit the changes
    conn.commit()
    cursor.close()


def get_latest_row(table_name):
    cursor = conn.cursor()

    # Select the most recent row based on timestamp
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 1")

    row = cursor.fetchone()
    cursor.close()

    return row

def get_latest_ohlcv(table_name):
    cursor = conn.cursor()

    # Select the most recent row's OHLCV data based on timestamp
    cursor.execute(f"""
        SELECT timestamp, open_value, high_value, low_value, close_value, volume 
        FROM {table_name} 
        ORDER BY timestamp DESC LIMIT 1
    """)

    ohlcv_data = cursor.fetchone()
    cursor.close()

    return ohlcv_data

if __name__ == '__main__':
    # connection parameters
    params = {
        "database": "Database",
        "user": "wheels",
        "password": "Brewen12",
        "host": "localhost",  # or the IP of your DB
        "port": "5432"  # default postgres port
    }

    # create a connection
    conn = psycopg2.connect(**params)