import copy

from torch import nn


def get_deep_clones(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
