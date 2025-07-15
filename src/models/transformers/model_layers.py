import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from models.transformers.embeding import DataEmbedding
from models.transformers.self_attention_layers import AttentionLayer, FullAttention, get_mask_pam, refer_points, \
    RegularMask


class PyraformerConvLayer(nn.Module):
    """
    A Pyraformer convolutional layer.

    This layer applies a 1D convolution with a window size of `window_size` to the input tensor `x`. The stride is also
    set to `window_size`, which means that the output tensor will have a lower temporal resolution than the input. The
    output is then passed through batch normalization and the ELU activation function.

    Args:
        c_in (int): The number of input channels.
        window_size (int): The size of the sliding window for the convolution operation.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, c_in, window_size):
        super(PyraformerConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck_Construct(nn.Module):
    """
    A bottleneck convolutional layer for the Pyraformer transformer.
    This layer implements a bottleneck convolutional CSCM (channel-split context modeling) operation.

    Args:
        d_model (int): The number of channels in the input and output tensors.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the convolution operation(s).
            If an integer is provided, the same window size is used for all convolutional layers.
            If a list of integers is provided, each convolutional layer uses a different window size.
        d_inner (int): The dimensionality of the intermediate tensor produced by the first linear layer.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                PyraformerConvLayer(d_inner, window_size),
                PyraformerConvLayer(d_inner, window_size),
                PyraformerConvLayer(d_inner, window_size)
            ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(PyraformerConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)
        return all_inputs


class PositionwiseFeedForward(nn.Module):
    """
    A Two-layer position-wise feed-forward neural network for the Pyraformer transformer.

    Args:
        d_in (int): The number of input and output channels.
        d_hid (int): The number of channels in the hidden layer.
        dropout (float): The dropout probability.
        normalize_before (bool): If True, apply layer normalization before the feed-forward network; if False, apply
            layer normalization after the feed-forward network.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class PyraformerEncoderLayer(nn.Module):
    """
    A single layer of the Pyraformer encoder.

    Args:
        d_model (int): The number of channels in the input and output tensors.
        d_inner (int): The number of channels in the intermediate tensor of the position-wise feed-forward network.
        n_head (int): The number of attention heads to use in the self-attention mechanism.
        dropout (float): The dropout probability.
        normalize_before (bool): If True, apply layer normalization before the self-attention mechanism; if False,
            apply layer normalization after the self-attention mechanism.

    Returns:
        Tensor: The output tensor of the layer.
    """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1, normalize_before=True):
        super(PyraformerEncoderLayer, self).__init__()

        self.slf_attn = AttentionLayer(
            FullAttention(mask_flag=True, attention_dropout=dropout, output_attention=False),
            d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        attn_mask = RegularMask(slf_attn_mask)
        enc_output, _ = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class PyraformerEncoder(nn.Module):
    """
    An encoder model with a Pyraformer self-attention mechanism.

    Args:
        enc_in (int): The number of input channels.
        seq_len (int): The length of the input sequence.
        d_model (int): The number of channels in the input and output tensors.
        n_heads (int): The number of attention heads to use in the self-attention mechanism.
        e_layers (int): The number of Pyraformer encoder layers to use.
        d_ff (int): The number of channels in the intermediate tensor of the position-wise feed-forward network.
        dropout (float): The dropout probability.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the bottleneck convolutional CSCM
            layer. If an integer is provided, the same window size is used for all convolutional layers. If a list of
            integers is provided, each convolutional layer uses a different window size.
        inner_size (int): The dimensionality of the intermediate tensor produced by the bottleneck linear layer.

    Returns:
        Tensor: The output tensor of the model.
    """

    def __init__(self, enc_in, seq_len, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, window_size=None, inner_size=4):
        super().__init__()

        if window_size is None:
            window_size = [4, 4]

        d_bottleneck = d_model // 4

        self.mask, self.all_size = get_mask_pam(seq_len, window_size, inner_size)

        self.indexes = refer_points(self.all_size, window_size)
        self.layers = nn.ModuleList([
            PyraformerEncoderLayer(d_model, d_ff, n_heads, dropout=dropout,
                                   normalize_before=False) for _ in range(e_layers)
        ])  # naive pyramid attention

        self.enc_embedding = DataEmbedding(
            enc_in, d_model, dropout)
        self.conv_layers = Bottleneck_Construct(
            d_model, window_size, d_bottleneck)

    def forward(self, x_enc, x_mark_enc):
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(
            0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        return seq_enc


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initializes a new instance of the EncoderLayer class,
        which applies a self-attention mechanism and a feedforward neural network (FFNN) to its input.

        Args:
        - attention (nn.Module): An attention module to use in the layer
        - d_model (int): The input and output dimensionality of the layer
        - d_ff (int or None): The hidden dimensionality of the FFNN, which defaults to 4 times the input dimensionality
                               if None is specified.
        - dropout (float): defaults to 0.1.
        - activation (str): The activation function either "relu" or "gelu".
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class ConvLayer(nn.Module):
    """
        A convolutional layer with batch normalization, ELU activation, and max pooling.

        Args:
        - c_in (int): The number of input channels.
    """

    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Encoder(nn.Module):
    """
    A multi-layer transformer encoder module that combines attention and convolution layers.

    Args:
    - attn_layers (list of EncoderLayer): A list of EncoderLayer instances that make up the attention layers.
    - conv_layers (list of ConvLayer or None): A list of ConvLayer instances that make up the convolution layers
    - norm_layer (nn.Module or None): A normalization layer to apply to the output of the encoder, or None if not used.
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Decoder layer of a transformer model, composed of self-attention and cross-attention sublayers.

    Args:
        self_attention (nn.Module): Self-attention layer.
        cross_attention (nn.Module): Cross-attention layer.
        d_model (int): Dimensionality of the model.
        d_ff (int, optional): Dimensionality of the feedforward network. Defaults to 4 * d_model.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        activation (str, optional): "relu" or "gelu". Defaults to "relu".
    Outputs:
        x (torch.Tensor): Output tensor of shape [batch_size, sequence_length, d_model].
    """

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    A decoder model for Transformer-based sequence-to-sequence architectures.

    Args:
        layers (list): A list of DecoderLayer objects that make up the layers of the decoder.
        norm_layer (nn.Module, optional): A normalization layer to apply after the decoder layers.
        projection (nn.Module, optional): A projection layer to apply to the final output of the decoder.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
