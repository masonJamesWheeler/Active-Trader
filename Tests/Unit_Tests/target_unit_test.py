import numpy as np
import pandas as pd
from io import StringIO
import sys

from Data.data import get_and_process_data

sys.path.append('../')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

if __name__ == '__main__':
    AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
    tickers = ["AAPL", "BA"]
    interval = '1min'
    threshhold = 0.01
    window_size = 30
    years = 2
    months = 12

    X_train, Y_train, X_test, Y_test = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
    # Convert Y_train and Y_test from one-hot encoding to single integer labels
    Y_train_max = np.argmax(Y_train, axis=1)
    Y_test_max = np.argmax(Y_test, axis=1)

    # Combine X_train and X_test, and Y_train and Y_test
    X_all = np.concatenate((X_train, X_test), axis=0)
    Y_all_max = np.concatenate((Y_train_max, Y_test_max), axis=0)

    plt.figure(figsize=(15, 5))
    plt.plot(X_all[:, -1, 4], label='Close')
    plt.scatter(np.where(Y_all_max == 1)[0], X_all[np.where(Y_all_max == 1)[0], -1, 4], marker='^', s=100, c='g', label='buy')
    plt.scatter(np.where(Y_all_max == 2)[0], X_all[np.where(Y_all_max == 2)[0], -1, 4], marker='v', s=100, c='r', label='sell')
    plt.legend()
    plt.show()
