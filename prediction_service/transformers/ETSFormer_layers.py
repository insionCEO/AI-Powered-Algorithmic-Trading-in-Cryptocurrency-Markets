import math

import torch
import torch.fft as fft
from einops import rearrange, reduce, repeat
from scipy.fftpack import next_fast_len
from torch import nn
import torch.nn.functional as F


class Transform:
    """
    Initializes the Transform object with a given sigma value.  used to apply a transformation to an input tensor.
    The transformation involves adding jitter, scaling, and shift to the input,
    which helps to make it more robust to small variations.

    Args:
        sigma: A float representing the amount of jitter, scale, and shift to be applied to the input.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


def conv1d_fft(f, g, dim=-1):
    """
    Computes the one-dimensional convolution of two input tensors using the Fast Fourier Transform (FFT) algorithm.

    The convolution is computed along a specified dimension of the input tensors.

    Args:
        f: A PyTorch tensor representing the first input tensor.
        g: A PyTorch tensor representing the second input tensor.
        dim: An integer representing the dimension along which to compute the convolution. Default is -1.

    Returns:
        A PyTorch tensor representing the result of the convolution.
    """
    N = f.size(dim)
    M = g.size(dim)

    fast_len = next_fast_len(N + M - 1)

    F_f = fft.rfft(f, fast_len, dim=dim)
    F_g = fft.rfft(g, fast_len, dim=dim)

    F_fg = F_f * F_g.conj()
    out = fft.irfft(F_fg, fast_len, dim=dim)
    out = out.roll((-1,), dims=(dim,))
    idx = torch.as_tensor(range(fast_len - N, fast_len)).to(out.device)
    out = out.index_select(dim, idx)

    return out


class ExponentialSmoothing(nn.Module):
    """
    A PyTorch module that computes the exponential smoothing of an input tensor
     using the Exponential Smoothing algorithm.

    Args:
        dim: An integer representing the dimension of the input tensors.
        nhead: An integer representing the number of attention heads to use.
        dropout: A float representing the dropout rate to use. Default is 0.1.
        aux: A boolean indicating whether to use auxiliary input. Default is False.
    """

    def __init__(self, dim, nhead, dropout=0.1, aux=False):
        super().__init__()
        self._smoothing_weight = nn.Parameter(torch.randn(nhead, 1))
        self.v0 = nn.Parameter(torch.randn(1, 1, nhead, dim))
        self.dropout = nn.Dropout(dropout)
        if aux:
            self.aux_dropout = nn.Dropout(dropout)

    def forward(self, values, aux_values=None):
        b, t, h, d = values.shape

        init_weight, weight = self.get_exponential_weight(t)
        output = conv1d_fft(self.dropout(values), weight, dim=1)
        output = init_weight * self.v0 + output

        if aux_values is not None:
            aux_weight = weight / (1 - self.weight) * self.weight
            aux_output = conv1d_fft(self.aux_dropout(aux_values), aux_weight)
            output = output + aux_output

        return output

    def get_exponential_weight(self, T):
        # Generate array [0, 1, ..., T-1]
        powers = torch.arange(T, dtype=torch.float, device=self.weight.device)

        # (1 - \alpha) * \alpha^t, for all t = T-1, T-2, ..., 0]
        weight = (1 - self.weight) * (self.weight ** torch.flip(powers, dims=(0,)))

        # \alpha^t for all t = 1, 2, ..., T
        init_weight = self.weight ** (powers + 1)

        return rearrange(init_weight, 'h t -> 1 t h 1'), \
            rearrange(weight, 'h t -> 1 t h 1')

    @property
    def weight(self):
        return torch.sigmoid(self._smoothing_weight)


class Feedforward(nn.Module):
    """
    A PyTorch module that implements a feedforward neural network used for ETSFormer.

    Args:
        d_model: An integer representing the number of features in the input tensor.
        dim_feedforward: An integer representing the dimension of the feedforward network.
        dropout: A float representing the dropout rate to use. Default is 0.1.
        activation: A string representing the activation function to use. Default is 'sigmoid'.
    """

    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='sigmoid'):
        # Implementation of Feedforward model
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)


class GrowthLayer(nn.Module):
    """
    A PyTorch module that implements the Growth Layer in the ETSFormer model,
     designed to capture growth patterns in the input time-series data.

    Args:
        d_model: An integer representing the number of features in the input tensor.
        nhead: An integer representing the number of attention heads to use.
        d_head: An integer representing the number of features per attention head. Defaults to None.
        dropout: A float representing the dropout rate to use. Default is 0.1.
    """

    def __init__(self, d_model, nhead, d_head=None, dropout=0.1):
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead

        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)

        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def forward(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        b, t, d = inputs.shape
        values = self.in_proj(inputs).view(b, t, self.nhead, -1)
        values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)
        out = torch.cat([repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), out], dim=1)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return self.out_proj(out)


class FourierLayer(nn.Module):
    """
    A PyTorch module that applies a Fourier Transform to the input tensor.
    The Fourier Transform is used to decompose the input time-series data into its frequency components.
    The magnitude-based selection is used to select the top-k frequency components that contain the most information
    about the time-series data.

    Args:
        d_model: An integer representing the number of features in the input tensor.
        pred_len: An integer representing the number of time steps to predict.
        k: An integer representing the number of frequency components to select. Defaults to None.
           If not provided, k is set to 2 * d_model // 3.
        low_freq: An integer representing the minimum frequency component to consider. Defaults to 1.
    """

    def __init__(self, d_model, pred_len, k=None, low_freq=1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        # index_tuple = tuple([i.to('cuda') for i in index_tuple])
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        # f = f.to('cuda')
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple


class LevelLayer(nn.Module):
    """
    The LevelLayer takes the level, growth, and seasonality embeddings as inputs, and produces the level embedding
    for the next time step. The level embedding represents the baseline value of the time series.

    Parameters:
        d_model (int): The input size of the module.
        c_out (int): The output size of the module.
        dropout (float): The dropout probability to be applied in the ExponentialSmoothing module.
    """

    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level, growth, season):
        b, t, _ = level.shape
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season).view(b, t, self.c_out, 1)
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return out


class DampingLayer(nn.Module):
    """
    A module that applies a damping factor to the input sequence.

    Args:
    - pred_len (int): The length of the sequence to be predicted.
    - nhead (int): The number of heads in the multi-head attention layer.
    - dropout (float): The dropout rate.
    """

    def __init__(self, pred_len, nhead, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.nhead = nhead
        self._damping_factor = nn.Parameter(torch.randn(1, nhead))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = repeat(x, 'b 1 d -> b t d', t=self.pred_len)
        b, t, d = x.shape

        powers = torch.arange(self.pred_len).to(self._damping_factor.device) + 1
        powers = powers.view(self.pred_len, 1)
        damping_factors = self.damping_factor ** powers
        damping_factors = damping_factors.cumsum(dim=0)
        x = x.view(b, t, self.nhead, -1)
        x = self.dropout(x) * damping_factors.unsqueeze(-1)
        return x.view(b, t, d)

    @property
    def damping_factor(self):
        return torch.sigmoid(self._damping_factor)
