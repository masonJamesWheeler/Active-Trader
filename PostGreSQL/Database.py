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
    cursor = None
    values_str = None
    try:
        cursor = conn.cursor()
    except Exception as e:
        print(f"An error occurred while creating cursor: {e}")
        return

    try:
        # Check the number of rows
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

        # If the row count exceeds the limit, delete the oldest row
        if row_count >= max_rows:
            cursor.execute(f"DELETE FROM {table_name} WHERE timestamp = (SELECT MIN(timestamp) FROM {table_name})")
    except Exception as e:
        print(f"An error occurred while managing table rows: {e}")
        return

    try:
        # Convert tensor to numpy if needed
        if isinstance(values, torch.Tensor):
            values = values.numpy()

        if isinstance(action, torch.Tensor):
            action = action.item()

        if isinstance(shares, torch.Tensor):
            shares = shares.item()
    except Exception as e:
        print(f"An error occurred while converting data: {e}")
        return

    try:
        # Prepare the values
        values_str = ', '.join(map(str, values.tolist()))
        values_str += f', {action}, {shares}'

        # Generate the timestamp
        timestamp = datetime.now()

        # Insert the new row
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

        # Commit the changes
        conn.commit()
    except Exception as e:
        print(f"An error occurred while inserting data into the database: {e}")
        print(f"Values string was: {values_str}")
    finally:
        cursor.close()


def get_latest_row(table_name):
    cursor = None
    try:
        cursor = conn.cursor()
    except Exception as e:
        print(f"An error occurred while creating cursor: {e}")
        return None

    try:
        # Select the most recent row based on timestamp
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
    except Exception as e:
        print(f"An error occurred while fetching the latest row: {e}")
        row = None
    finally:
        cursor.close()

    return row


def get_latest_n_rows(table_name, n):
    cursor = None
    try:
        cursor = conn.cursor()
    except Exception as e:
        print(f"An error occurred while creating cursor: {e}")
        return []

    try:
        # Select the most recent n rows based on timestamp
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {n}")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"An error occurred while fetching the latest {n} rows: {e}")
        rows = []
    finally:
        cursor.close()

    return rows


def get_latest_ohlcv(table_name):
    cursor = None
    try:
        cursor = conn.cursor()
    except Exception as e:
        print(f"An error occurred while creating cursor: {e}")
        return None

    try:
        # Select the most recent row's OHLCV data based on timestamp
        cursor.execute(f"""
            SELECT timestamp, open_value, high_value, low_value, close_value, volume 
            FROM {table_name} 
            ORDER BY timestamp DESC LIMIT 1
        """)
        ohlcv_data = cursor.fetchone()
    except Exception as e:
        print(f"An error occurred while fetching the latest OHLCV data: {e}")
        ohlcv_data = None
    finally:
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