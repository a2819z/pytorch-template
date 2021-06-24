import torch.nn as nn

from .modules.block import SkipBlock, ConvBlock, UpBlock, DownBlock, AdaIN


def parse_model(cfg) -> list:
    # m: module, args: module args, n: depth
    layers = []
    for i, (m, args, n) in enumerate(cfg):
        m = eval(m) if isinstance(m, str) else m
        skip_flag = True if isinstance(m, SkipBlock) else False

        if args == None:
            m_ = m()
        else:
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)

        layers.append(m_)

    return nn.ModuleList(*layers), skip_flag
