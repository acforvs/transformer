import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """ Since our model contains no recurrence and no convolution, in order for the model to make use of the
    order of the sequence, we must inject some information about the relative or absolute position of the
    tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
    bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
    as the embeddings, so that the two can be summed. There are many choices of positional encodings,
    learned and fixed.
    """

    def __init__(self, d_model: int, p_dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_t = torch.pow(
            10000.,
            -torch.arange(0, d_model, 2, dtype=torch.float) / d_model,
        )
        p_table = torch.zeros(max_len, d_model)
        p_table[:, 0::2] = torch.sin(position * div_t)
        p_table[:, 1::2] = torch.cos(position * div_t)
        p_table = p_table.unsqueeze(0)
        self.register_buffer('p_table', p_table)

    def forward(self, x):
        return self.dropout(
            x + self.p_table[:x.shape[1]]
        )
