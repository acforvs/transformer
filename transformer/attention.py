import torch
from torch import nn
import numpy as np

from helpers import get_deep_clones


class MultiHeadedAttention(nn.Module):
    """ MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V),
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    https://arxiv.org/pdf/1706.03762.pdf see pages 4 and 5
    We use d_k = d_v = d_model / h
    """
    def __init__(self, model_dim, n_heads, p_dropout=0.1, save_attn_weights=False):
        super(MultiHeadedAttention, self).__init__()
        assert model_dim % n_heads == 0
        self.d_k = model_dim // n_heads
        self.n_heads = n_heads
        # one net for each of query, key and value
        self.qkv_nns = get_deep_clones(
            nn.Linear(model_dim, model_dim), 3
        )
        self.out_nn = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.save_attn_weights = save_attn_weights
        self.attn_weights = None

    def _attention(self, query, key, value, mask):
        # shape is (n_batches, n_heads, tokens len, d_k)
        scores = torch.matmul(query, torch.transpose(key, -2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores.masked_fill(mask == 0, -np.inf)
        ps = self.softmax(scores)
        # dropout is not used in the original paper, see page 4, figure 2 (left)
        # https://arxiv.org/pdf/1706.03762.pdf
        ps = self.dropout(ps)
        return torch.matmul(ps, value), ps

    def forward(self, query, key, value, mask):
        n_batches = query.shape[0]
        # reshaping a tensor from (n_batches, tokens len, n_heads * d_k) to (n_batches, tokens len, n_heads, d_k) to
        #   (n_batches, n_heads, tokens len, d_k) which is the shape for an input tensor for self._attention
        query, key, value = [
            qkv_nn(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
            for qkv_nn, x in zip(self.qkv_nns, (query, key, value))
        ]
        x, attn_weights = self._attention(query, key, value, mask)
        if self.save_attn_weights:
            self.attn_weights = attn_weights

        # reshaping to (n_batches, tokens len, n_heads * d_k)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)

        return self.out_nn(x)
