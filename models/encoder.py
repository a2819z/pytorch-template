from functools import partial
import torch.nn as nn

from models.modules.block import ConvBlock, ResBlock, SkipBlock, DownBlock


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.module, skip_flag = encoder_builder(cfg)

        if skip_flag:
            self.skip_list = []

    def forward(self, x):
        for layer in self.module:
            x = layer(x)

        if hasattr(self, "skip_list"):
            return x, self.skip_list

        return x

    def get_skip(self):
        for layer in self.module:
            if isinstance(layer, SkipBlock):
                self.skip_list.append(layer.skip)


def encoder_builder(cfg):
    ConvBlk = partial(ConvBlock, norm=cfg.norm, activ=cfg.activ, pad_type=cfg.pad_type)
    ResBlk = partial(ResBlock, norm=cfg.norm, activ=cfg.activ, pad_type=cfg.pad_type)

    layers = []
    skip_flag = False
    for i, (m, args, n) in enumerate(cfg.struct):
        m = eval(m) if isinstance(m, str) else m
        skip_flag |= issubclass(m, SkipBlock)

        if args == None:
            m_ = m()
            continue
        elif isinstance(m, ConvBlock):
            m = ConvBlk
        elif isinstance(m, ResBlock):
            m = ResBlk

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        layers.append(m_)

    return nn.ModuleList(layers), skip_flag
