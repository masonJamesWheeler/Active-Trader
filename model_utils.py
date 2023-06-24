import tensorflow as tf
from keras import layers
from tensorflow import keras
import json
import os

def load_model_from_checkpoint(trial_dir, model_creation_func):
    # Load hyperparameters
    hp_filepath = os.path.join(trial_dir, 'trial.json')
    hp = load_hyperparameters_from_json(hp_filepath)

    # Create a model with the loaded hyperparameters
    model = model_creation_func(hp)

    # Create a Checkpoint object with the model as the target to restore
    checkpoint = tf.train.Checkpoint(model=model)

    # Use the Checkpoint object to restore the model weights.
    checkpoint.restore(tf.train.latest_checkpoint(trial_dir))

    return model

def dense_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(30, 14)))
    model.add(keras.layers.Dense(units=hp['dense_1_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_1']))
    model.add(keras.layers.Dense(units=hp['dense_2_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_2']))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])
    return model

def conv_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(filters=hp['conv_1_filter'],
                                  kernel_size=hp['conv_1_kernel'],
                                  activation='relu',
                                  input_shape=(30, 14)))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_1_size']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=hp['dense_1_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_1']))
    model.add(keras.layers.Dense(units=hp['dense_2_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_2']))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])
    return model

def lstm_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=hp['lstm_1_units'],
                                activation='relu',
                                return_sequences=True,
                                input_shape=(30, 14)))
    model.add(keras.layers.MaxPooling1D(pool_size=hp['pool_1_size']))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=hp['dense_1_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_1']))
    model.add(keras.layers.Dense(units=hp['dense_2_units'], activation='relu'))
    model.add(keras.layers.Dropout(rate=hp['dropout_2']))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(hp['learning_rate']),
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
    for _ in range(hp['num_transformer_layers']):
        x = transformer_encoder(
            x,
            head_size=hp['head_size'],
            num_heads=hp['num_heads'],
            ff_dim=hp['ff_dim'],
            dropout=hp['dropout']
        )
    outputs = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(hp['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', multi_label=True, label_weights=None)])
    return model

def load_hyperparameters_from_json(trial_json_path):
    with open(trial_json_path, 'r') as f:
        trial_data = json.load(f)

    # Extract the 'values' field which contains the optimal hyperparameters
    hp_values = trial_data['hyperparameters']['values']

    # We'll remove any hyperparameters that are not used in the model creation function
    irrelevant_hp_keys = ['tuner/epochs', 'tuner/initial_epoch', 'tuner/bracket', 'tuner/round', 'tuner/trial_id']
    hp = {k: v for k, v in hp_values.items() if k not in irrelevant_hp_keys}

    return hp
