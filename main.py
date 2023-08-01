import torch
import numpy as np
import matplotlib.pyplot as plt
from Data.data import get_and_process_data, get_all_months

data, scaled_data, scaler = get_and_process_data('AMZN', '1min', 128, '2023-07')

fig, ax = plt.subplots(2, 1)
ax[0].plot(data[:, 3])
ax[1].plot(scaled_data[:, 3])
plt.show()