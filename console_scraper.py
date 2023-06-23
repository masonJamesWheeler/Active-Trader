# import os
# import json
# import csv
# import glob
#
#
# def process_trial_files(directory):
#     csv_data = []
#     trial_files = glob.glob(os.path.join(directory, 'trial_*/trial.json'))
#
#     for trial_file in trial_files:
#         with open(trial_file, 'r') as file:
#             data = json.load(file)
#
#             row = {}
#             row['trial_id'] = data.get('trial_id')
#             row.update(data.get('hyperparameters', {}).get('values', {}))
#             row.update({
#                 'accuracy':
#                     data.get('metrics', {}).get('metrics', {}).get('accuracy', {}).get('observations', [{}])[0].get(
#                         'value', [0])[0],
#                 'auc':
#                     data.get('metrics', {}).get('metrics', {}).get('auc', {}).get('observations', [{}])[0].get('value',
#                                                                                                                [0])[0],
#                 'time': os.path.getmtime(trial_file),  # replace this with your own time data if available
#                 'row_label': data.get('trial_id')  # add row_label that mirrors trial_id
#             })
#
#             csv_data.append(row)
#
#     return csv_data
#
#
# def write_to_csv(csv_data, output_file):
#     if not csv_data:
#         print('No data to write.')
#         return
#
#     # Get all field names from the data
#     fieldnames = list(set().union(*[d.keys() for d in csv_data]))
#
#     # Write data to CSV
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in csv_data:
#             writer.writerow(row)
#
#
# # process the trial files
# csv_data = process_trial_files('my_dir/Stock_Trading_dense')
#
# # write the data to CSV
# write_to_csv(csv_data, 'output.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_hyperparam_effect(df, hyperparameter):
    """
    This function plots the average effect of changing a hyper-parameter on
    the change of AUC and accuracy.
    """
    hyperparam = ""
    # Change the name of the hyperparameter to be more readable
    if hyperparameter == 'dense_1_units':
        hyperparam = 'Number of Units in First Dense Layer'
    elif hyperparameter == 'dropout_1':
        hyperparam = 'Dropout Rate in First Dropout Layer'
    elif hyperparameter == 'dense_2_units':
        hyperparam = 'Number of Units in Second Dense Layer'
    elif hyperparameter == 'dropout_2':
        hyperparam = 'Dropout Rate in Second Dropout Layer'
    elif hyperparameter == 'learning_rate':
        hyperparam = 'Learning Rate'

    # set the style of the plot to be white grid for better contrast in black and white
    sns.set_style("whitegrid")

    # set the font to DejaVu Serif
    plt.rcParams['font.family'] = 'DejaVu Serif'

    plt.figure(figsize=(10, 8), dpi = 400)

    # calculate the means of accuracy and AUC for each unique value of the hyperparameter
    means = df.groupby(hyperparameter)[['accuracy', 'auc']].mean().reset_index()

    # plot the change in accuracy as the hyperparameter changes
    plt.subplot(2, 1, 1)
    sns.lineplot(data=means, x=hyperparameter, y='accuracy', marker='o', color='black')
    plt.title(f'Effect of the {hyperparam} on Accuracy')
    plt.xlabel(hyperparameter)
    plt.ylabel('Average Accuracy')

    # plot the change in AUC as the hyperparameter changes
    plt.subplot(2, 1, 2)
    sns.lineplot(data=means, x=hyperparameter, y='auc', marker='o', color='black')
    plt.title(f'Effect of {hyperparam} on AUC Score')
    plt.xlabel(hyperparameter)
    plt.ylabel('Average AUC')

    plt.tight_layout()
#   Save the image to the Dense_Images folder
    plt.savefig(f'Dense_Images/{hyperparameter}.png')

# Assume df is your DataFrame
df = pd.read_csv('output.csv')

# Call the function for each hyperparameter
for hyper in ['dense_1_units', 'dropout_1', 'dense_2_units', 'dropout_2', 'learning_rate']:
    plot_hyperparam_effect(df, hyper)
