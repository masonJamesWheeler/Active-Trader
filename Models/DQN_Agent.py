import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MultiheadAttention


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, architecture, dense_layers, dense_size, dropout_rate):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.dense_size = dense_size
        self.dense_layers_num = dense_layers
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.num_actions = num_actions


        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after RNN layer 1
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after RNN layer 2


        # Dynamic construction of dense layers
        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()  # Dropout after each dense layer
        for i in range(dense_layers):
            if i == 0:
                self.dense_layers.append(nn.Linear(hidden_size, dense_size))
            else:
                self.dense_layers.append(nn.Linear(dense_size // (2 ** (i - 1)), dense_size // (2 ** i)))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(dense_size // (2 ** (dense_layers - 1)), num_actions)

    def load_weights(self, target, ticker, dense_layers, dense_size, hidden_size, dropout_rate, feature_size, num_actions):
        if target:
            if os.path.exists(f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}_target.pth'):
                self.load_state_dict(torch.load(f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}_target.pth'))
        else:
            if os.path.exists(f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}.pth'):
                self.load_state_dict(torch.load(f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}.pth'))

    def save_weights(self, target, ticker, dense_layers, dense_size, hidden_size, dropout_rate, feature_size, num_actions):
        if target:
            torch.save(self.state_dict(), f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}_target.pth')
        else:
            torch.save(self.state_dict(), f'./Weights/DQN{ticker}_{dense_layers}_{dense_size}_{hidden_size}_{dropout_rate}_{feature_size}_{num_actions}.pth')

    def forward(self, x, hidden_state1, hidden_state2):
        if isinstance(self.rnn1, nn.LSTM):
            h1, c1 = hidden_state1
            h2, c2 = hidden_state2
            x, (h1, c1) = self.rnn1(x, (h1, c1))
            x = self.dropout1(x)  # Apply dropout
            x, (h2, c2) = self.rnn2(x, (h2, c2))
            x = self.dropout2(x)  # Apply dropout
            hidden_state1 = (h1, c1)
            hidden_state2 = (h2, c2)
        else:
            x, hidden_state1 = self.rnn1(x, hidden_state1)
            x = self.dropout1(x)  # Apply dropout
            x, hidden_state2 = self.rnn2(x, hidden_state2)
            x = self.dropout2(x)  # Apply dropout

        x = hidden_state2[-1]  # Using the last hidden state of the second GRU

        # Pass through dynamic dense layers
        for i, (dense_layer, dropout_layer) in enumerate(zip(self.dense_layers, self.dropout_layers)):
            x = F.relu(dense_layer(x))
            x = dropout_layer(x)  # Apply dropout

        x = self.output_layer(x)

        return x, hidden_state1, hidden_state2

    def init_hidden(self, batch_size):
        if isinstance(self.rnn1, nn.LSTM):
            return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)), \
                (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))
        else:
            return torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)


class MetaModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=10, output_size=1, window_size=10):
        super(MetaModel, self).__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Reshaping the input to fit the model
        x = x.view(-1, self.window_size, 2)

        lstm_out, _ = self.lstm(x)
        # Getting the last output of the sequence
        lstm_out = lstm_out[:, -1, :]

        x = torch.sigmoid(self.fc(lstm_out))
        return x

    def load_weights(self, input_size, hidden_layer_size, output_size, window_size):
        if os.path.exists(f'./Weights/MetaModel_{input_size}_{hidden_layer_size}_{output_size}_{window_size}.pth'):
            self.load_state_dict(torch.load(f'./Weights/MetaModel_{input_size}_{hidden_layer_size}_{output_size}_{window_size}.pth'))

    def save_weights(self, input_size, hidden_layer_size, output_size, window_size):
        torch.save(self.state_dict(), f'./Weights/MetaModel_{input_size}_{hidden_layer_size}_{output_size}_{window_size}.pth')

def update_Q_values(batch, Q_network, target_network, optimizer, architecture, gamma=0.99):
    """
    Update the Q values and perform a backward pass.

    Args:
        batch: A batch of experiences containing states, actions, rewards, next states and hidden states.
        Q_network: The current Q network model.
        target_network: The target Q network model.
        optimizer: The optimizer used to update the weights of the Q network.
        architecture (str): The architecture of the Q network (e.g., 'LSTM').
        gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.

    Returns:
        None
    """
    # Convert the batch data into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    def process_batch(hidden_state):
        """
        Process the hidden states from the batch.

        Args:
            hidden_state: A list of hidden states.

        Returns:
            A tensor containing the processed hidden states.
        """
        # Check if hidden state is a tuple (for LSTM)
        if isinstance(hidden_state[0], tuple):
            return (torch.stack([x[0] for x in hidden_state]).squeeze().unsqueeze(0),
                    torch.stack([x[1] for x in hidden_state]).squeeze().unsqueeze(0))
        else:
            return torch.stack(hidden_state).squeeze().unsqueeze(0)

    # Process the hidden states
    hidden_state1_batch = process_batch(batch.hidden_state1)
    hidden_state2_batch = process_batch(batch.hidden_state2)
    next_hidden_state1_batch = process_batch(batch.next_hidden_state1)
    next_hidden_state2_batch = process_batch(batch.next_hidden_state2)

    # Compute current Q values
    current_Q_values, _, _ = Q_network(state_batch, hidden_state1_batch, hidden_state2_batch)

    # Gather the Q values for the actions taken
    current_Q_values = current_Q_values.gather(1, action_batch.view(-1, 1)).squeeze()

    # Compute Q values for next states
    next_Q_values, _, _ = target_network(next_state_batch, next_hidden_state1_batch, next_hidden_state2_batch)
    # For LSTM architecture, remove unnecessary dimension

    next_Q_values = next_Q_values.max(1)[0].detach()

    # Compute the target Q values
    target_Q_values = reward_batch + (gamma * next_Q_values)
    # Compute the loss between current and target Q values
    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

    # Zero the gradients
    optimizer.zero_grad()
    # Perform a backward pass to compute gradients
    loss.backward()
    # Update the weights of the Q network
    optimizer.step()
