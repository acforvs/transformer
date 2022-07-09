import numpy as np
from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_model: int, d_vocab: int):
        super(Embedding, self).__init__()
        self.embeddings = nn.Embedding(d_vocab, d_model)
        self.dim = d_model

    def forward(self, tokens):
        # (n_batches, tokens len) to (n_batches, tokens len, d_model) which is then submitted to MHA
        return self.embeddings(tokens) * np.sqrt(self.dim)
