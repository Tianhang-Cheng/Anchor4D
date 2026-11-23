# Frozen point-tracking feature encoder (ConvNeXt stem + 1x1 projection) used for the correlation reader.
# Architecture and weights from AllTracker (https://github.com/aharley/alltracker), Copyright (c) 2025 Adam W. Harley,
# MIT License. Only the feature encoder (`cnn` + `dot_conv`) is kept.
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class CNBlock2d(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(0., 'row')
        self.final = nn.Identity()

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return self.final(result)


class ConvNeXt(nn.Module):
    """ConvNeXt-tiny stages 1-3 at stride 8 (stage 3 downsampling replaced by a dilated 3x3)."""

    def __init__(self):
        super().__init__()
        norm = partial(LayerNorm2d, eps=1e-6)
        layers = [Conv2dNormActivation(3, 96, kernel_size=4, stride=4, padding=0, norm_layer=norm,
                                       activation_layer=None, bias=True)]
        for cin, cout, n, down in ((96, 192, 3, True), (192, 384, 3, False), (384, None, 9, False)):
            layers.append(nn.Sequential(*[CNBlock2d(cin) for _ in range(n)]))
            if cout is not None:
                conv = nn.Conv2d(cin, cout, kernel_size=2, stride=2) if down else \
                    nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=2, dilation=2)
                layers.append(nn.Sequential(norm(cin), conv))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class TrackerFeatures(nn.Module):
    """[B, T, S, S, 3] uint8 frames -> [B, T, 256, S/8, S/8] features (bf16 autocast)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(ConvNeXt(), nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255, persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255, persistent=False)

    @torch.no_grad()
    def forward(self, pixels):
        b, t, h, w, _ = pixels.shape
        x = (pixels.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float() - self.mean) / self.std
        with torch.autocast('cuda', torch.bfloat16):
            f = self.encoder(x)
        return f.reshape(b, t, *f.shape[1:])
