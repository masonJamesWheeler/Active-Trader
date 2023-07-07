import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, architecture, dense_layers, dense_size, dropout_rate):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size

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
                self.dense_layers.append(nn.Linear(dense_size // (2 ** (i-1)), dense_size // (2 ** i)))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(dense_size // (2 ** (dense_layers-1)), num_actions)

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


