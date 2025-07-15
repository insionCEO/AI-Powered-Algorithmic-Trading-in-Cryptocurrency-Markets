import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEmbedding(nn.Module):
    """


    Args:
        d_model (int): The dimensionality of the input/output vectors.
        max_len (int, optional): The maximum sequence length that the positional
            encoding should support. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # Create a tensor with positions [0, 1, ..., max_seq_len-1]
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # Compute the denominator of the positional encoding formula
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Compute the sine and cosine positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a new dimension to the tensor to make its shape (1, max_seq_len, d_model).
        # This ensures that the positional embedding tensor can be used with batches of input data.
        pe = pe.unsqueeze(0)

        # Register the tensor as a buffer so that it will be saved with the model and can be accessed later.
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    This class implements the token embedding layer using a 1D convolutional neural network.

    Args:
        c_in (int): The number of input channels (i.e., the size of the vocabulary).
        d_model (int): The dimensionality of the output vectors.
    """

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    This class implements the fixed positional embedding layer, where the embedding
    weights are computed using sine and cosine functions of the position.

    Args:
        c_in (int): The number of input channels
        d_model (int): The dimensionality of the output vectors.
    """

    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='Sec'):
        super(TemporalEmbedding, self).__init__()

        # Define the number of unique values for each time-related feature
        num_minutes = 60
        num_hours = 24
        num_weekdays = 7
        num_days = 32
        num_months = 13

        # Choose the appropriate embedding type based on the input argument
        if embed_type == 'fixed':
            Embedding = FixedEmbedding
        else:
            Embedding = nn.Embedding

        # Initialize the embeddings for each time-related feature
        if freq == 'Sec':
            self.second_embed = Embedding(num_minutes, d_model)
        if freq == 'Min':
            self.minute_embed = Embedding(num_minutes, d_model)
        self.hour_embed = Embedding(num_hours, d_model)
        self.weekday_embed = Embedding(num_weekdays, d_model)
        self.day_embed = Embedding(num_days, d_model)
        self.month_embed = Embedding(num_months, d_model)

    def forward(self, x):
        # Convert the input tensor to a long tensor
        x = x.long()

        # Retrieve the embeddings for each time-related feature
        if hasattr(self, 'second_embed'):
            second_x = self.second_embed(x[:, :, 5])
        else:
            second_x = 0.
        if hasattr(self, 'minute_embed'):
            minute_x = self.minute_embed(x[:, :, 4])
        else:
            minute_x = 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # Sum the embeddings for all time-related features
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # create a token embedding for the input values
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # create a positional embedding for the input values
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # create a temporal embedding for the time features
        # 'fixed' type of embedding is used by default
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        # create a dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # apply token embedding to the input values and add it to the positional embedding and temporal embedding
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        # apply dropout regularization and return the output
        return self.dropout(x)
