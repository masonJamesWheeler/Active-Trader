import requests
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, MaxPooling1D, Dropout, Flatten, Dense
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from data import get_and_process_data
from sklearn.utils import class_weight

def search_model(x_train, y_train, x_test, y_test):
    y_train_classes = np.argmax(y_train, axis=1)
    # Define the base neural network model
    weights = class_weight.compute_class_weight(y=y_train_classes, class_weight='balanced', classes=np.unique(y_train_classes))
    # Convert class weights to dictionary
    class_weights = dict(enumerate(weights))

    def dense_model(hp):
        model = keras.models.Sequential()

        # Add a Flatten layer as first layer to handle 2D input
        model.add(keras.layers.Flatten(input_shape=(30, 14)))

        # Add a Dense layer with hyperparameterized number of units and ReLU activation
        model.add(keras.layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a Dropout layer with hyperparameterized dropout rate
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a second Dense layer
        model.add(keras.layers.Dense(units=hp.Int('dense_2_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a second Dropout layer
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a Dense output layer with softmax activation for multi-class classification
        model.add(keras.layers.Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])

        return model

    def conv_model(hp):
        model = keras.models.Sequential()

        # Add a Conv1D layer with hyperparameterized number of filters and kernel_size
        model.add(keras.layers.Conv1D(filters=hp.Int('conv_1_filter', min_value=32, max_value=256, step=32),
                                      kernel_size=hp.Int('conv_1_kernel', min_value=3, max_value=15, step=2),
                                      activation='relu',
                                      input_shape=(30, 14)))

        # Add a MaxPooling1D layer
        model.add(keras.layers.MaxPooling1D(pool_size=hp.Int('pool_1_size', min_value=2, max_value=5, step=1)))

        # Add a Flatten layer as first layer to handle 2D input
        model.add(keras.layers.Flatten())

        # Add a Dense layer with hyperparameterized number of units and ReLU activation
        model.add(keras.layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a Dropout layer with hyperparameterized dropout rate
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a second Dense layer
        model.add(keras.layers.Dense(units=hp.Int('dense_2_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a second Dropout layer
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a Dense output layer with softmax activation for multi-class classification
        model.add(keras.layers.Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])

        return model

    def lstm_model(hp):
        model = keras.models.Sequential()

        # Add a LSTM layer with hyperparameterized number of units
        model.add(keras.layers.LSTM(units=hp.Int('lstm_1_units', min_value=32, max_value=256, step=32),
                                    activation='relu',
                                    return_sequences=True,  # make sure LSTM returns output for each time step
                                    input_shape=(30, 14)))

        # Add a MaxPooling1D layer
        model.add(keras.layers.MaxPooling1D(pool_size=hp.Int('pool_1_size', min_value=2, max_value=5, step=1)))

        # Add a Flatten layer as first layer to handle 2D input
        model.add(keras.layers.Flatten())

        # Add a Dense layer with hyperparameterized number of units and ReLU activation
        model.add(keras.layers.Dense(units=hp.Int('dense_1_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a Dropout layer with hyperparameterized dropout rate
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a second Dense layer
        model.add(keras.layers.Dense(units=hp.Int('dense_2_units', min_value=32, max_value=512, step=32),
                                     activation='relu'))

        # Add a second Dropout layer
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        # Add a Dense output layer with softmax activation for multi-class classification
        model.add(keras.layers.Dense(3, activation='softmax'))

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])

        return model

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def transformer_model(hp):
        inputs = keras.Input(shape=(30, 14))
        x = inputs
        for _ in range(hp.Int('num_transformer_layers', 1, 3)):
            x = transformer_encoder(
                x,
                head_size=hp.Int('head_size', min_value=32, max_value=128, step=32),
                num_heads=hp.Int('num_heads', 2, 8, step=2),
                ff_dim=hp.Int('ff_dim', min_value=32, max_value=128, step=32),
                dropout=hp.Float('dropout', min_value=0.0, max_value=0.3, step=0.1)
            )
        outputs = layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])
        return model

    # Create an EarlyStopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model_types = ['dense', 'conv', 'lstm', 'transformer']

    # Loop over model types
    for model_type in model_types:

        # Select the appropriate model based on model_type
        if model_type == 'dense':
            model_fn = dense_model
        elif model_type == 'conv':
            model_fn = conv_model
        elif model_type == 'lstm':
            model_fn = lstm_model
        elif model_type == 'transformer':
            model_fn = transformer_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        # Instantiate the tuner
        tuner = kt.Hyperband(model_fn,
                             objective=kt.Objective("val_auc", direction="max"),
                             max_epochs=10,
                             factor=3,
                             directory='my_dir',
                             project_name=f'Stock_Trading_{model_type}')  # Make project_name unique for each model

        # Perform hyperparameter search
        tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early], class_weight=class_weights)

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete for {model_type} model. Here are the optimal hyperparameters:
        {best_hps.values}.
        """)

    print("All hyperparameter searches are complete.")

if __name__ == "__main__":
    AlphaVantage_Free_Key = "A5QND05S0W7CU55E"
    tickers = ["AAPL", "AMZN", "META", "NFLX", "NVDA", "TSLA"]
    interval = '1min'
    threshhold = 0.01
    window_size = 30
    years = 2
    months = 12

    x_train, y_train, x_test, y_test = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold, window_size, years, months)
    # Print what proportion of the y_test is 0
    print("Proportion of 0s in y_test:", np.sum(y_test == 0) / len(y_test))
    search_model(x_train, y_train, x_test, y_test)
