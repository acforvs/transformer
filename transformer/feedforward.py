from torch import nn


class PositionWiseFeedForwardNN(nn.Module):
    """ In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
    connected feed-forward network, which is applied to each position separately and identically. This
    consists of two linear transformations with a ReLU activation in between.
    While the linear transformations are the same across different positions, they use different parameters
    from layer to layer.
    """

    def __init__(self, d_model: int, inner_scaler=4, p_dropout=None):
        super(PositionWiseFeedForwardNN, self).__init__()
        d_inner = d_model * inner_scaler
        self.w1 = nn.Linear(d_model, d_inner)
        self.w2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(p=p_dropout) if p_dropout else None
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.w1(x))
        return self.w2(self.dropout(x) if self.dropout else x)
