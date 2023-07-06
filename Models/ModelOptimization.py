# We will use this file to store all the parameters that we will use to optimize our models

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





