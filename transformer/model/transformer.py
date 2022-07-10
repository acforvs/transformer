from torch import nn

from transformer.model.embedding import Embedding
from transformer.model.positional import PositionalEncoding
from transformer.model.attention import MultiHeadedAttention
from transformer.model.feedforward import PositionWiseFeedForwardNN
from transformer.model.encoder_decoder import Encoder, EncoderLayer, Decoder, DecoderLayer, DecoderGenerator


class Transformer(nn.Module):
    def __init__(
            self, d_model=512, n_heads=8, d_vocab_inp=1000, d_vocab_out=1000,
            n_layers=6, p_dropout=0.1, save_attn_weights=False
    ):
        super(Transformer, self).__init__()
        self.inp_embedding = Embedding(d_model, d_vocab_inp)
        self.out_embedding = Embedding(d_model, d_vocab_out)

        self.inp_positional_enc = PositionalEncoding(d_model, p_dropout)
        self.out_positional_enc = PositionalEncoding(d_model, p_dropout)

        multi_head_attn = MultiHeadedAttention(d_model, n_heads, p_dropout, save_attn_weights)
        feedforward_nn = PositionWiseFeedForwardNN(d_model, inner_scaler=4)

        self.encoder = Encoder(
            EncoderLayer(d_model, multi_head_attn, feedforward_nn, p_dropout),
            n_layers,
        )
        self.decoder = Decoder(
            DecoderLayer(d_model, multi_head_attn, feedforward_nn, p_dropout),
            n_layers,
        )
        self.decoder_generator = DecoderGenerator(d_model, d_vocab_out)

    def forward(self, x, y, mask_x, mask_y):
        return self.decode(
            self.encode(x, mask_x), y, mask_x, mask_y,
        )

    def encode(self, x, mask):
        x = self.inp_embedding(x)
        x = self.inp_positional_enc(x)
        return self.encoder(x, mask)

    def decode(self, x, y, mask_x, mask_y):
        y = self.out_embedding(y)
        y = self.out_positional_enc(y)

        log_ps = self.decoder_generator(
            self.decoder(y, x, mask_y, mask_x)
        )
        return log_ps.reshape(-1, log_ps.shape[-1])
