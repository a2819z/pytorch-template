from functools import partial

import torch.nn as nn

from models.modules import parse_layer_from_cfg
from models.modules.block import ConvBlock, ResBlock, SkipBlock, AdaIN


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.module, skip_flag = decoder_builder(cfg)

        if cfg.out == "sigmoid":
            self.out = nn.Sigmoid()
        elif cfg.out == "tanh":
            self.out = nn.Tanh()
        else:
            raise ValueError(cfg.out)

    def forward(self, x, styles, interpolation_weights=None):
        if interpolation_weights is None:
            interpolation_weights = [1 / len(styles)] * len(styles)

        skip_idx = 0
        for layer in self.module:
            if isinstance(layer, AdaIN):
                self.interpolate(layer, x, styles, interpolation_weights, skip_idx)
                skip_idx += 1

            x = layer(x)

        return x

    def interpolate(self, layer, x, styles, interpolation_weights, skip_idx):
        transformed_features = []
        for style, interpolation_weight in zip(styles, interpolation_weights):
            transformed_features.append(
                layer(x, style[skip_idx]) * interpolation_weight
            )

        return sum(transformed_features)

    def get_skip(self):
        for layer in self.module:
            if isinstance(layer, SkipBlock):
                self.skip_list.append(layer.skip)


def decoder_builder(cfg):
    ConvBlk = partial(ConvBlock, norm=cfg.norm, activ=cfg.activ, pad_type=cfg.pad_type)
    ResBlk = partial(ResBlock, norm=cfg.norm, activ=cfg.activ, pad_type=cfg.pad_type)

    layers = []
    for i, (m, args, n) in enumerate(cfg):
        m = eval(m) if isinstance(m, str) else m
        skip_flag = True if isinstance(m, SkipBlock) else False

        if args == None:
            m_ = m()
            continue
        elif isinstance(m, ConvBlock):
            m = ConvBlk
        elif isinstance(m, ResBlock):
            m = ResBlk

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        layers.append(m_)

    return nn.ModuleList(*layers), skip_flag
