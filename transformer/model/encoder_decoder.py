import copy

from torch import nn

from transformer.model.helpers import get_deep_clones
from transformer.model.attention import MultiHeadedAttention
from transformer.model.feedforward import PositionWiseFeedForwardNN


class SublayerConnection(nn.Module):
    """ Residual Dropout
    We apply dropout [33] to the output of each sub-layer, before it is added to the
    sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    P_drop = 0.1.
    """

    def __init__(self, d_model: int, p_dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        """ http://nlp.seas.harvard.edu/2018/04/03/attention.html
        we use: normalization -> sublayer -> dropout -> addition
        paper uses: sublayer -> dropout -> addition -> normalization
        """
        normalized = sublayer(self.norm(x))
        return x + self.dropout(normalized)


# ========== ENCODER GOES HERE ==========s

class EncoderLayer(nn.Module):
    """ Each layer has two sub-layers.
    The first is a multi-head self-attention mechanism,
    and the second is a simple, positionwise fully connected feed-forward network
    """

    def __init__(
            self, d_model: int, multi_head_attn: MultiHeadedAttention,
            feed_forward_nn: PositionWiseFeedForwardNN, p_dropout=0.1,
    ):
        super(EncoderLayer, self).__init__()
        self.dim = d_model
        self.sublayers = get_deep_clones(
            SublayerConnection(d_model, p_dropout),
            2,  # num_sublayers
        )
        self.multi_head_attn = multi_head_attn
        self.feed_forward_nn = feed_forward_nn

    def forward(self, x, mask):
        sublayer_1, sublayer_2 = self.sublayers
        x = sublayer_1(
            x,
            lambda inp: self.multi_head_attn(query=inp, key=inp, value=inp, mask=mask)
        )
        return sublayer_2(x, self.feed_forward_nn)


class Encoder(nn.Module):
    """ The encoder is composed of a stack of N = 6 identical layers.
    """

    def __init__(self, layer: EncoderLayer, num_layers=6):
        super(Encoder, self).__init__()
        self.layers = get_deep_clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ========== DECODER GOES HERE ==========

class DecoderLayer(nn.Module):
    """ In addition to the two sub-layers in each encoder layer,
    the decoder inserts a third sub-layer, which performs multi-head
    attention over the output of the encoder stack.
    """

    def __init__(
            self, d_model: int, multi_head_attn: MultiHeadedAttention,
            feed_forward_nn: PositionWiseFeedForwardNN, p_dropout=0.1
    ):
        super(DecoderLayer, self).__init__()
        self.dim = d_model
        self.sublayers = get_deep_clones(
            SublayerConnection(d_model, p_dropout),
            3,  # num_sublayers
        )
        self.multi_head_attn = copy.deepcopy(multi_head_attn)
        # performs multi-head attention over the output of the encoder stack
        self.enc_multi_head_attn = copy.deepcopy(multi_head_attn)
        self.feed_forward_nn = feed_forward_nn

    def forward(self, x, memory, inp_mask, out_mask):
        # memory will be an output of the encoder later on
        mem = memory
        sublayer_1, sublayer_2, sublayer_3 = self.sublayers
        x = sublayer_1(
            x, lambda inp: self.multi_head_attn(query=inp, key=inp, value=inp, mask=out_mask)
        )
        # https://arxiv.org/pdf/1706.03762.pdf, page 4, right pic
        x = sublayer_2(
            x, lambda inp: self.enc_multi_head_attn(query=inp, key=mem, value=mem, mask=inp_mask)
        )
        return sublayer_3(x, self.feed_forward_nn)


class Decoder(nn.Module):
    """ The decoder is also composed of a stack of N = 6 identical layers.
    """

    def __init__(self, layer: DecoderLayer, num_layers=6):
        super(Decoder, self).__init__()
        self.layers = get_deep_clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.dim)

    def forward(self, x, memory, inp_mask, out_mask):
        for layer in self.layers:
            x = layer(x, memory, inp_mask, out_mask)
        return self.norm(x)


class DecoderGenerator(nn.Module):
    def __init__(self, d_model: int, d_vocab: int):
        super(DecoderGenerator, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_vocab),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.feed_forward(x)
