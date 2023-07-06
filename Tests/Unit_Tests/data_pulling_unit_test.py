import unittest
import pandas as pd
import numpy as np
from io import StringIO
import requests
from sklearn.preprocessing import MinMaxScaler
import os
import sys

from Data.Data import get_stock_data, get_all_data

sys.path.append('../')


base_url = 'https://www.alphavantage.co/query?'
AlphaVantage_Free_Key = "A5QND05S0W7CU55E"

class TestGetStockData(unittest.TestCase):
    def setUp(self):
        self.symbol = 'AAPL'
        self.interval = '1min'
        self.api_key = "A5QND05S0W7CU55E"

    def test_get_stock_data(self):
        # Call the function to be tested
        df = get_stock_data(symbol=self.symbol, interval=self.interval, api_key=self.api_key, adjusted=True, extended_hours=True)

        # Check if the returned object is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check if the DataFrame is not empty
        self.assertGreater(len(df), 0)

        # Check if the DataFrame has the correct columns
        expected_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        self.assertTrue(all([col in df.columns for col in expected_columns]))

        # Check if the DataFrame values are all finite
        self.assertTrue(df.replace([np.inf, -np.inf], np.nan).dropna().equals(df))

        # Check if 'time' column is in correct format (YYYY-MM-DD HH:MM)
        self.assertTrue(all(pd.to_datetime(df['time'], errors='coerce').notna()))

class TestGetAllData(unittest.TestCase):

    def setUp(self):
        self.symbol = 'AAPL'
        self.interval = '1min'
        self.api_key = 'A5QND05S0W7CU55E'
        self.window_size = 30

    def test_get_all_data_return_type(self):
        df, indices = get_all_data(self.symbol, self.interval, self.api_key, self.window_size)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(isinstance(indices, dict))

    def test_get_all_data_columns(self):
        df, indices = get_all_data(self.symbol, self.interval, self.api_key, self.window_size)
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'smawindow', 'emawindow', 'sma50', 'ema50', 'sma200', 'ema200', 'rsi', 'rsi2', 'time']
        self.assertTrue(all(column in df.columns for column in expected_columns))

    def test_get_all_data_no_nan_values(self):
        df, indices = get_all_data(self.symbol, self.interval, self.api_key, self.window_size)
        self.assertFalse(df.isnull().values.any())

    def test_get_all_data_scaled_values(self):
        df, indices = get_all_data(self.symbol, self.interval, self.api_key, self.window_size)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        pd.testing.assert_frame_equal(df, df_scaled)

if __name__ == "__main__":
    # create a TestSuite object
    suite = unittest.TestSuite()

    # add test classes to the TestSuite
    suite.addTest(unittest.makeSuite(TestGetStockData))
    suite.addTest(unittest.makeSuite(TestGetAllData))

    # create a TextTestRunner with verbosity=2 (for detailed results)
    runner = unittest.TextTestRunner(verbosity=2)

    # run the suite using the runner
    result = runner.run(suite)

    print("\nRan {} tests. {} passed, {} failed.".format(result.testsRun,
                                                         result.testsRun - len(result.failures) - len(result.errors),
                                                         len(result.failures)))


