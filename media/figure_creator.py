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