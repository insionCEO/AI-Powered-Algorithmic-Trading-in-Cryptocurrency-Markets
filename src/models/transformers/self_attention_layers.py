import math

import torch
import torch.nn as nn
import numpy as np

from math import sqrt


def get_mask_pam(input_size, window_size, inner_size):
    """
    Get the attention mask for the PAM-Naive layer in the Pyraformer transformer.
    The mask defines the connections between tokens in different layers and optimizes the attention mechanism by
    limiting the allowed connections.

    Args:
        input_size (int): The size of the input sequence.
        window_size (list of int): The window size of each layer in the PAM-Naive layer.
        inner_size (int): The size of the inner window for the intra-scale mask.

    Returns:
        mask (torch.Tensor, bool): A boolean tensor representing the attention mask for the PAM-Naive layer.
        all_size (list of int): A list of integers representing the size of all the layers in the PAM-Naive layer.
    """

    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length)

    # get intra-scale mask.
    # This mask is responsible for allowing attention within a certain range in the same layer.
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask.
    # This mask is responsible for connecting tokens from one layer to another in the pyramid architecture
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + \
                        (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (
                                     start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(all_sizes, window_size):
    """
    Computes a mapping between input tokens and their corresponding tokens in each layer of the pyramid architecture
    used by the Pyraformer encoder.

    Args:
        all_sizes (list[int]): A list of the number of tokens in each layer of the pyramid architecture.
        window_size (int or list[int]): The size(s) of the sliding window(s) for the bottleneck convolutional CSCM
            layer. If an integer is provided, the same window size is used for all convolutional layers. If a list of
            integers is provided, each convolutional layer uses a different window size.

    Returns:
        Tensor: The mapping between input tokens and their corresponding tokens in each layer of the pyramid
        architecture.
    """
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes))

    # loop through all the tokens in the input sequence and for each token iterate
    # through all the layers in the pyramid architecture
    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + \
                           min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


class RegularMask():
    """
    A utility class for handling attention masks in the Pyraformer architecture.

    Args:
        mask (Tensor): The attention mask tensor.
    """
    def __init__(self, mask):
        self._mask = mask.unsqueeze(1)

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask:
    """
    Generates an upper-triangular binary mask to prevent the model from attending to future time-steps during
    self-attention mechanism.

    Args:
    - B (int): Batch size.
    - L (int): Sequence length.
    - device (str or torch.device): Device on which to create the mask.

    Attributes:
    - mask (torch.Tensor): A tensor of shape (B, 1, L, L) representing the binary mask.
    """

    def __init__(self, B, L, device="cpu"):
        # Create a mask shape of size [B, 1, L, L] where B is the batch size, L is the sequence length
        mask_shape = [B, 1, L, L]

        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Probability mask to be applied to attention scores, where each position i in the mask
        represents whether the ith query is allowed to attend to positions > i in the values.

        Args:

        - B (int): Batch size.
        - H (int): Number of attention heads.
        - L (int): Length of the input queries.
        - index (torch.Tensor): Indices of top-k queries for each batch and head, of shape (B, H, c*ln(L_Q)).
        - scores (torch.Tensor): Attention scores tensor of shape (B, H, L_Q, L_V).
        - device (str or torch.device): Device to be used for tensor operations.

        Returns:
        _mask (torch.Tensor): Probability mask tensor of shape (B, H, L_Q, L_V).
        """
        upper_triangular_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)

        expanded_mask = upper_triangular_mask[None, None, :].expand(B, H, L, scores.shape[-1])

        indicator = expanded_mask[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)

        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        """
        FullAttention module performs scaled dot-product attention between the input queries, keys, and values.
        used in the original transformer

        Args:
        - mask_flag (bool): Whether to apply the attention mask. Default: True.
        - scale (float or None): Scale factor for the attention scores. Default: None.
        - attention_dropout (float): Dropout probability for the attention weights. Default: 0.1.
        - output_attention (bool): Whether to return the attention scores. Default: False.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # Get the shape of the input tensors.
        B, L, H, E = queries.shape  # Batch size, sequence length, number of heads, embedding dimension.
        _, S, _, D = values.shape  # Batch size, sequence length, number of heads, hidden size.
        scale = self.scale or 1. / np.sqrt(E)  # Scaling factor.

        # Compute the attention scores between the queries and keys using matrix multiplication.
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply a mask to the attention scores if the mask flag is set.
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Apply dropout and softmax to the scaled attention scores.
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Compute the weighted sum of the values using the attention scores.
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # If output_attention is True, return both the output tensor and the attention scores.
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        Initializes a new instance of the ProbAttention class.

        Args:
        - mask_flag (bool): A boolean that indicates whether to use masking for self-attention or not.
        - factor (int): Controls the number of sampled keys to use in the calculation.
        - scale (float or None): Scaling factor for the dot product attention. If None, the default scale is 1/sqrt(D).
        - attention_dropout (float): Dropout probability for the attention mechanism.
        - output_attention (bool): A boolean that indicates whether to output the attention matrix or not.
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Computes the dot product attention between the queries and values tensors, with sparse top-k selection.

        Args:
        - Q (torch.Tensor): Input queries tensor of shape (B, H, L_Q, D).
        - K (torch.Tensor): Input keys tensor of shape (B, H, L_K, D).
        - sample_k (int): Number of keys to sample per query.
        - n_top (int): Number of top-k queries to select.

        Returns:
        - Q_K (torch.Tensor): Attention output tensor of shape (B, H, n_top, D).
        - M_top (torch.Tensor): Indices of top-k queries for each batch and head, of shape (B, H, n_top).
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # expand the K tensor to match the dimensions of Q
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # randomly sample K indices
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]

        # calculate the dot product attention scores
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # compute sparsity measurement M
        M_top = M.topk(n_top, sorted=False)[1]  # select top-k values of M

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        # calculate the dot product attention using Q_reduce and K
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        Computes the initial context vector from the input values matrix.

        Args:
        - V (torch.Tensor): Input values matrix of shape (B, H, L, D).
        - L_Q (int): Length of the input queries.

        Returns:
        - contex (torch.Tensor): Initial context vector of shape (B, H, L_Q, D).
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        Updates the context vector with the attended values.

        Args:
        - context_in (torch.Tensor): Input context vector of shape (B, H, L_Q, D).
        - V (torch.Tensor): Input values matrix of shape (B, H, L_V, D).
        - scores (torch.Tensor): Attention scores tensor of shape (B, H, L_Q, L_V).
        - index (torch.Tensor): Indices of top-k queries for each batch and head, of shape (B, H, c*ln(L_Q)).
        - L_Q (int): Length of the input queries.
        - attn_mask (torch.Tensor): Attention mask tensor of shape (B, H, L_Q, L_V).

        Returns:
        - context_in (torch.Tensor): Updated context vector of shape (B, H, L_Q, D).
        - attns (torch.Tensor or None): Attention probability tensor of shape (B, H, L_Q, L_V),
                                        or None if `output_attention` is False.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        Initializes a new instance of the AttentionLayer class which creates the multi-head self-attention mechanism

        Args:
        - attention (nn.Module): An attention module to use in the layer.
        - d_model (int): The input and output dimensionality of the layer.
        - n_heads (int): The number of attention heads.
        - d_keys (int or None): The dimensionality of the query and key vectors.
                                  If None, d_model // n_heads is used.
        - d_values (int or None): The dimensionality of the value vectors.
                                  If None, d_model // n_heads is used.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
