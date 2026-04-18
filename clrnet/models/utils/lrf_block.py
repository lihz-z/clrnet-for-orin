import math

import torch
import torch.nn as nn


class FrequencyGate(nn.Module):
    """Suppress rain-dominated high-frequency responses while preserving edges."""

    def __init__(self, channels, threshold=0.25):
        super().__init__()
        self.threshold = threshold
        init_gate = torch.full((channels,), 0.0)
        self.gate_factor = nn.Parameter(init_gate)

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = torch.fft.rfft2(x.float(), norm='ortho')

        freq_y = torch.fft.fftfreq(h, device=x.device).view(-1, 1)
        freq_x = torch.fft.rfftfreq(w, device=x.device).view(1, -1)
        freq_dist = torch.sqrt(freq_y**2 + freq_x**2)
        freq_dist = freq_dist / (freq_dist.max() + 1e-6)
        high_mask = (freq_dist > self.threshold).to(x_fft.real.dtype)
        high_mask = high_mask.view(1, 1, h, w // 2 + 1)

        gate = torch.sigmoid(self.gate_factor).view(1, c, 1, 1)
        freq_weight = 1.0 - high_mask + high_mask * gate
        filtered = torch.fft.irfft2(x_fft * freq_weight, s=(h, w), norm='ortho')

        return filtered.to(x.dtype) - x


def _depthwise_branch(channels, kernel_size, dilation=1):
    padding = ((kernel_size[0] - 1) // 2 * dilation,
               (kernel_size[1] - 1) // 2 * dilation)
    return nn.Sequential(
        nn.Conv2d(channels,
                  channels,
                  kernel_size,
                  padding=padding,
                  dilation=dilation,
                  groups=channels,
                  bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class DirectionalAttention(nn.Module):
    """Enhance elongated lane structures with lightweight directional filters."""

    def __init__(self, channels, bins=4):
        super().__init__()
        branch_builders = [
            lambda: _depthwise_branch(channels, (1, 5)),
            lambda: _depthwise_branch(channels, (5, 1)),
            lambda: nn.Sequential(
                _depthwise_branch(channels, (1, 3)),
                _depthwise_branch(channels, (3, 1)),
            ),
            lambda: nn.Sequential(
                _depthwise_branch(channels, (3, 1)),
                _depthwise_branch(channels, (1, 3)),
            ),
            lambda: _depthwise_branch(channels, (3, 3), dilation=2),
            lambda: _depthwise_branch(channels, (3, 3), dilation=3),
        ]
        bins = max(2, min(bins, len(branch_builders)))
        self.dir_convs = nn.ModuleList(
            [branch_builders[i]() for i in range(bins)])
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels * bins, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        dir_feats = [conv(x) for conv in self.dir_convs]
        attn = self.attn_conv(torch.cat(dir_feats, dim=1))
        return x * attn


class LRFBlock(nn.Module):
    """Frequency gate + directional attention + lightweight residual fusion."""

    def __init__(self,
                 channels,
                 freq_gate_threshold=0.25,
                 direction_bins=4,
                 res_scale_init=0.1,
                 enable_freq=True,
                 enable_dir=True):
        super().__init__()
        self.enable_freq = enable_freq
        self.enable_dir = enable_dir
        if not (self.enable_freq or self.enable_dir):
            raise ValueError('At least one branch must be enabled in LRFBlock.')
        self.freq = FrequencyGate(channels, threshold=freq_gate_threshold) \
            if self.enable_freq else None
        self.dir = DirectionalAttention(channels, bins=direction_bins) \
            if self.enable_dir else None
        init = min(max(res_scale_init, 1e-4), 1 - 1e-4)
        self.res_scale = nn.Parameter(torch.tensor(math.log(init / (1 - init)),
                                                   dtype=torch.float32))

    def forward(self, x):
        fused = 0.0
        if self.enable_freq:
            fused = fused + self.freq(x)
        if self.enable_dir:
            fused = fused + self.dir(x)
        scale = torch.sigmoid(self.res_scale)
        out = x + scale * fused
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
