import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

# Read in the data
data = pd.read_csv('portfolio_values_1.csv')

# Replace the 'Step' column with a new column that just contains the row number
data['Step'] = range(len(data))

# Write the modified DataFrame back to the csv file
data.to_csv('v5.csv', index=False)

# Calculate the excess return, which we are calling "Alpha" for simplicity
data['Alpha'] = ((data['DQN Agent Portfolio Value'] - data['Buy and Hold Portfolio Value']) / data['Buy and Hold Portfolio Value']) * 100

# Create a dictionary to store data for each stock
stock_data = {}
stock_id = 0

# Create lists to store values for the current stock
current_stock_prices = []
actions = []
buy_and_hold_portfolio_values = []
dqn_agent_portfolio_values = []
alphas = []

# Iterate through rows in the dataframe
for i in range(len(data)):

    # If this is the first row or the 'Step' has decreased from the previous row
    if i == 0 or data.loc[i, 'Step'] < data.loc[i - 1, 'Step']:

        # If this is not the first row, save the lists to the dictionary
        if i != 0:
            stock_data[stock_id] = {
                'Current Stock Prices': current_stock_prices,
                'Actions': actions,
                'Buy and Hold Portfolio Values': buy_and_hold_portfolio_values,
                'DQN Agent Portfolio Values': dqn_agent_portfolio_values,
                'Alphas': alphas,
            }
            stock_id += 1

        # Start new lists for the new stock
        current_stock_prices = []
        actions = []
        buy_and_hold_portfolio_values = []
        dqn_agent_portfolio_values = []
        alphas = []

    # Append the values for the current row to the lists
    current_stock_prices.append(data.loc[i, 'Current Stock Price'])
    actions.append(data.loc[i, 'Action'])
    buy_and_hold_portfolio_values.append(data.loc[i, 'Buy and Hold Portfolio Value'])
    dqn_agent_portfolio_values.append(data.loc[i, 'DQN Agent Portfolio Value'])
    alphas.append(data.loc[i, 'Alpha'])

# Save the lists for the last stock to the dictionary
stock_data[stock_id] = {
    'Current Stock Prices': current_stock_prices,
    'Actions': actions,
    'Buy and Hold Portfolio Values': buy_and_hold_portfolio_values,
    'DQN Agent Portfolio Values': dqn_agent_portfolio_values,
    'Alphas': alphas,
}

# Loop through the dictionary and plot data for each stock
for stock_id, stock_values in stock_data.items():
    if stock_id == 0:
        stock_id = 'AAPL'
    # Research style plots
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'Serif', 'font.weight': 'semibold', 'axes.labelweight': 'semibold', 'axes.titleweight': 'semibold'})
    # Space the plots out vertically
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), dpi=800)
    # Make the line colors black
    axs[0].plot(stock_values['Current Stock Prices'], label='Current Stock Price', color='black')
    axs[1].plot(stock_values['Buy and Hold Portfolio Values'], label='Buy and Hold Portfolio Value', color='black')
    # make the DQN Agent's portfolio dashed
    axs[1].plot(stock_values['DQN Agent Portfolio Values'], label='DQN Agent Portfolio Value', color='#4338ca')
    # make the Alpha's portfolio

    # Make the Legends for All Plots
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")

    axs[0].set_title('Current Stock Price {}'.format(stock_id))
    axs[0].set_xlabel('Time Steps (Minutes)')
    axs[0].set_ylabel('Price (USD)')

    axs[1].set_title('Portfolio Value {}'.format(stock_id))
    axs[1].set_ylabel('Value (USD)')
    axs[1].set_xlabel('Time Steps (Minutes)')

    # Save each plot
    plt.savefig('./Results/Media/Results_{}.png'.format(stock_id))

final_hold_return = (data.loc[len(data) - 1, "Buy and Hold Portfolio Value"] - data.loc[0, "Buy and Hold Portfolio Value"]) / data.loc[0, "Buy and Hold Portfolio Value"]
final_agent_return = (data.loc[len(data) - 1, "DQN Agent Portfolio Value"] - data.loc[0, "DQN Agent Portfolio Value"]) / data.loc[0, "DQN Agent Portfolio Value"]
print("Final Alpha: ", data.loc[len(data) - 1, "Alpha"])
print("Final Buy and Hold Return: ", final_hold_return)
print("Final Agent Return: ", final_agent_return)


