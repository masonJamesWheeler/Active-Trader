import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

from data import get_and_process_data


AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
tickers = ["AAPL"]
interval = '1min'
threshhold = 0.01
window_size = 30
years = 2
months = 12

X_train, Y_train, X_test, Y_test = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
print(X_train.shape)
plt.figure(figsize=(15, 5))
plt.plot(X_train[ :, -1, 3], label='Close')
# plot where the buy and sell signals are as bubbles on the price
plt.scatter(np.where(Y_train == 1)[0], X_train[np.where(Y_train == 1)[0], -1, 3], marker='^', s=100, c='g', label='buy')
plt.scatter(np.where(Y_train == 2)[0], X_train[np.where(Y_train == 2)[0], -1, 3], marker='v', s=100, c='r', label='sell')
plt.show()

