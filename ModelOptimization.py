import numpy as np           # Handle matrices
import matplotlib.pyplot as plt # Display graphs

from collections import deque

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from StockEnvironment import StockEnvironment, ReplayMemory, EpsilonGreedyStrategy
from data import get_and_process_data
from StockEnvironment import StockEnvironment
import random
from collections import namedtuple
import warnings

def squared_portfolio_difference(post_value, pre_value):
    sign = 1 if post_value - pre_value > 0 else -1
    return sign * (post_value - pre_value) ** 2

def portfolio_difference(post_value, pre_value):
    return post_value - pre_value

architectures = ["RNN", "GRU", "LSTM"]
WindowSizes = [60, 120, 180, 240]
HiddenSizes = [32, 64, 128, 256]
Dense_Sizes = [32, 64, 128, 256]
Dense_Layers = [1, 2, 3, 4]

reward_functions = ['squared', 'linear']





