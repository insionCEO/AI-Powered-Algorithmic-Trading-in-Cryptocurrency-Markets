import torch
from torch import nn

from models.transformers.embeding import DataEmbedding
from models.transformers.model_layers import Encoder, EncoderLayer, ConvLayer, Decoder, DecoderLayer, PyraformerEncoder
from models.transformers.self_attention_layers import ProbAttention, FullAttention, AttentionLayer


class Informer(nn.Module):
    """
    Transformer-based encoder-decoder architecture for time series forecasting.

    Args:
        enc_in (int): Number of input features for the encoder.
        dec_in (int): Number of input features for the decoder.
        c_out (int): Number of output classes or features.
        seq_len (int): Length of input sequence.
        label_len (int): Length of label sequence.
        out_len (int): Length of output sequence.
        factor (int, optional): Factor for dimensionality reduction in attention layers. Defaults to 5.
        d_model (int, optional): Number of hidden units in the model. Defaults to 512.
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        e_layers (int, optional): Number of layers for the encoder. Defaults to 3.
        d_layers (int, optional): Number of layers for the decoder. Defaults to 2.
        d_ff (int, optional): Number of units in the feed-forward layer. Defaults to 512.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        attn (str, optional): Type of attention mechanism. Either 'prob' (for prob sparse attention) or 'full'
                                (for full attention). Defaults to 'prob'.
        embed (str, optional): Type of data embedding. Either 'fixed' (for fixed position embeddings) or 'learned'
                                (for learned position embeddings). Defaults to 'fixed'.
        freq (str, optional): Time frequency of the input data. Either 'Min' (for Minutes), 'Sec' (for Seconds)
        activation (str, optional): Type of activation function. Either 'relu' or 'gelu' Defaults to 'gelu'.
        output_attention (bool, optional): Whether to return attention weights. Defaults to False.
        distil (bool, optional): Whether to use a distilled convolutional layer for downsampling in the encoder.
                                Defaults to True.
        device (torch.device, optional): Device to use for computations. Defaults to torch.device('cuda:0').
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='Sec', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class Transformer(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='full', embed='fixed', freq='Sec', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Transformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class Pyraformer(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity

    window_size: list, the downsample window size in pyramidal attention.
    inner_size: int, the size of neighbour attention
    """

    def __init__(self, enc_in, seq_len, out_len, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, window_size=None, inner_size=5):
        super(Pyraformer, self).__init__()
        if window_size is None:
            window_size = [4, 4]

        self.pred_len = out_len
        self.d_model = d_model

        self.encoder = PyraformerEncoder(enc_in, seq_len, d_model, n_heads, e_layers, d_ff,
                                         dropout, window_size, inner_size)

        self.projection = nn.Linear(
            (len(window_size) + 1) * self.d_model, self.pred_len * enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

