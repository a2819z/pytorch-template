import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.content_enc = Encoder(cfg.contnet_enc)
        self.style_enc = Encoder(cfg.style_enc)
        self.decoder = Decoder(cfg.decoder)

    def forward(self, content, styles, interpolation_weight=None):
        x = self.content_enc(content)

        style_features = []
        for style in styles:
            s, s_features = self.style_enc(style)
            s_features.reverse()  # Reordering style features for decoding
            s_features.append(s)
            style_features.append(s_features)

        x = self.decoder(x, style_features, interpolation_weight)

        return x
