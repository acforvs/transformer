from transformer import __version__
from transformer.model import Transformer


def test_version():
    assert __version__ == '0.1.0'


def test_transformer_init():
    assert Transformer()
