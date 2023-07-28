import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features)  # new shortcut connection
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        residual = self.shortcut(x)  # apply the shortcut connection to the input
        out = self.relu(self.linear1(x))
        out = self.dropout(self.linear2(out))
        out += residual
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_size, output_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else output_dim, output_dim) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, num_layers, output_dim):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ResidualBlock(input_size if i==0 else output_dim, output_dim) for i in range(num_layers)])
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.linear(x)

class FeatureProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super(FeatureProjection, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        # encoder_outputs shape: [batch_size, sequence_len, hidden_dim]
        energy = self.v(torch.tanh(self.W(encoder_outputs)))  # shape: [batch_size, sequence_len, 1]
        attention = F.softmax(energy, dim=1)  # shape: [batch_size, sequence_len, 1]
        return attention

class TemporalDecoder(nn.Module):
    def __init__(self, in_features, out_features, covariate_dim, hidden_dim, output_len):
        super(TemporalDecoder, self).__init__()
        self.attention = TemporalAttention(hidden_dim)
        self.residual_block = ResidualBlock(2 * hidden_dim, out_features)  # changed hidden_dim to output_dim
        self.covariate_projection = nn.Linear(covariate_dim, hidden_dim)  # changed out_features to hidden_dim
        self.output_len = output_len

    def forward(self, x, projected_covariates):
        outputs = torch.zeros(x.size(0), self.output_len, 5).to(x.device)  # initialize output tensor
        for t in range(self.output_len):
            attention = self.attention(x)
            context = torch.sum(attention * x, dim=1)  # shape: [batch_size, hidden_dim]
            context = context.unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
            projected_covariate = self.covariate_projection(projected_covariates[:, t, :]).unsqueeze(
                1)  # apply projection
            x = self.residual_block(
                torch.cat((context, projected_covariate), dim=-1))  # shape: [batch_size, 1, hidden_dim]
            outputs[:, t, :] = x.squeeze(1)
        return outputs


class TiDE(nn.Module):
    def __init__(self, input_size, num_encoder_layers, num_decoder_layers, output_dim, projected_dim):
        super(TiDE, self).__init__()
        self.feature_projection = FeatureProjection(input_size, projected_dim)
        self.encoder = Encoder(projected_dim, output_dim, num_encoder_layers)
        self.dense_decoder = Decoder(output_dim, num_decoder_layers, output_dim)
        self.temporal_decoder = TemporalDecoder(output_dim, output_dim, input_size, output_dim, 10)
        self.global_residual_connection = nn.Linear(input_size, output_dim)
        self.global_attention = TemporalAttention(output_dim)  # added this line

    def forward(self, x, covariates):
        projected_x = self.feature_projection(x)
        encoded_x = self.encoder(projected_x)
        decoded_x = self.dense_decoder(encoded_x)
        final_output = self.temporal_decoder(encoded_x, covariates)

        # apply global attention to reduce sequence length
        global_residual = self.global_residual_connection(x)
        attention = self.global_attention(global_residual)
        global_residual = torch.sum(attention * global_residual, dim=1)  # shape: [batch_size, hidden_dim]
        global_residual = global_residual.unsqueeze(1).expand(-1, 10, -1)  # shape: [batch_size, 10, hidden_dim]

        return final_output + global_residual

    def save_weights(self, ticker, epoch, live=True):
        if live:
            dir_path = f'Live_Weights'
        else:
            dir_path = f'BackTest_Weights'

        dir_path = f'{dir_path}/{ticker}'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), f'{dir_path}/TiDE_{epoch}.pth')

    def load_weights(self, ticker, live=True):
        if live:
            dir_path = f'Live_Weights'
        else:
            dir_path = f'BackTest_Weights'
        i = 0
        while os.path.isfile(f'{dir_path}/{ticker}/TiDE_{i}.pth'):
            i += 1
        if i > 0:
            self.load_state_dict(torch.load(f'{dir_path}/{ticker}/TiDE_{i - 1}.pth'))
        else:
            print(f"No weights found for {ticker}. Starting with random weights.")