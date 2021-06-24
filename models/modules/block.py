from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from .modules import spectral_norm


class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = "none"

        return dispatch_fn(key, *args)

    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        "none": nn.Identity,
        "in": partial(nn.InstanceNorm2d, affine=False),
        "bn": nn.BatchNorm2d,
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    return {"spectral": spectral_norm, "none": lambda x: x}[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d,
    }[pad_type.lower()]


class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        return x


class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return x


class SkipBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = None

    def forward(self, x):
        self.skip = x
        return x


class LinearBlock(nn.Module):
    """ pre-active linear block """

    def __init__(
        self,
        C_in,
        C_out,
        norm="none",
        activ="relu",
        bias=True,
        w_norm="none",
        dropout=0.0,
    ):
        super().__init__()
        activ = activ_dispatch(activ, norm)
        if norm.lower() == "bn":
            norm = nn.BatchNorm1d
        elif norm.lower() == "none":
            norm = nn.Identity
        else:
            raise ValueError(f"LinearBlock supports BN only (but {norm} is given)")
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return self.linear(x)


class ConvBlock(nn.Module):
    """ pre-active conv block """

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        activ="relu",
        bias=True,
        upsample=False,
        downsample=False,
        w_norm="none",
        pad_type="zero",
        dropout=0.0,
        size=None,
    ):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample

        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        padding=1,
        upsample=False,
        downsample=False,
        norm="none",
        w_norm="none",
        activ="relu",
        pad_type="zero",
        dropout=0.0,
        scale_var=False,
    ):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(
            C_in,
            C_out,
            kernel_size,
            1,
            padding,
            norm,
            activ,
            upsample=upsample,
            w_norm=w_norm,
            pad_type=pad_type,
            dropout=dropout,
        )
        self.conv2 = ConvBlock(
            C_out,
            C_out,
            kernel_size,
            1,
            padding,
            norm,
            activ,
            w_norm=w_norm,
            pad_type=pad_type,
            dropout=dropout,
        )

        # XXX upsample / downsample needs skip conv?
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, "skip"):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


class AdaIN(nn.Module):
    """
    AdaIN(Adaptive Instance Normalization) implementation
    ref: https://arxiv.org/abs/1703.06868
    """

    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style, eps=1e-8):
        b, c, h, w = content.size()

        style_mean = style.view(b, c, -1).mean(dim=2, keepdim=True)
        style_std = self.cal_std(style.view(b, c, -1), dim=2, keepdim=True, eps=eps)

        # Issue: https://github.com/pytorch/pytorch/issues/4320
        # tensor.std() has some error.
        content_std = self.cal_std(content.view(b, c, -1), dim=2, keepdim=True)
        content_mean = content.view(b, c, -1).mean(dim=2, keepdim=True)

        normalized = (content.view(b, c, -1) - content_mean) / (content_std + eps)
        stylized = (normalized * style_std + style_mean).view(b, c, h, w)

        return stylized

    def cal_std(self, tensor, dim, unbiased=True, keepdim=False, eps=1e-8):
        _var = tensor.var(dim=2, unbiased=unbiased) + eps
        _std = _var.sqrt()

        if keepdim:
            _std = _std.unsqueeze(dim)

        return _std

