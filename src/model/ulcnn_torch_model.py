"""PyTorch ULCNN (Ultra-Lightweight Complex Neural Network) for benchmark.

Faithful port of the Keras ULCNN architecture:
  Transpose → ComplexConv1D → ComplexBN → ReLU
  → 6× (SeparableConv1D stride=2 → BN → ReLU → ChannelShuffle)
  → ChannelAttention on each mobile unit
  → GAP of last 3 stages → Add fusion → Dense → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """Complex convolution: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # in_channels / out_channels refer to complex channels
        # internally we store separate real/imag kernels
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

    def forward(self, x):
        # x: (B, 2*C_in, L) where first C_in channels are real, next C_in are imag
        C = x.size(1) // 2
        x_r, x_i = x[:, :C], x[:, C:]
        out_r = self.conv_r(x_r) - self.conv_i(x_i)
        out_i = self.conv_r(x_i) + self.conv_i(x_r)
        return torch.cat([out_r, out_i], dim=1)


class ComplexBatchNorm1d(nn.Module):
    """Batch normalization applied independently to real and imag parts."""
    def __init__(self, num_complex_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_complex_channels * 2)

    def forward(self, x):
        return self.bn(x)


def channel_shuffle(x, groups=2):
    B, C, L = x.shape
    x = x.view(B, groups, C // groups, L)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, L)


class DWConvMobile(nn.Module):
    """Depthwise separable conv (stride=2) + BN + ReLU + ChannelShuffle."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        # SeparableConv1D = depthwise + pointwise
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=2, padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = channel_shuffle(x, groups=2)
        return x


class ChannelAttention(nn.Module):
    """Channel attention: GAP + GMP → shared MLP → sigmoid → scale."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, L)
        avg = x.mean(dim=2)   # (B, C)
        mx = x.amax(dim=2)    # (B, C)
        a = self.fc2(F.relu(self.fc1(avg)))
        m = self.fc2(F.relu(self.fc1(mx)))
        w = torch.sigmoid(a + m).unsqueeze(2)  # (B, C, 1)
        return x * w


class ULCNN(nn.Module):
    def __init__(self, num_classes=11, n_neuron=16, n_mobileunit=6, kernel_size=5):
        super().__init__()
        complex_ch = n_neuron          # 16 complex filters
        real_ch = complex_ch * 2       # 32 real channels

        # Initial complex conv: input has 1 complex channel (2 real: I, Q)
        self.complex_conv = ComplexConv1d(1, complex_ch, kernel_size, padding=kernel_size // 2)
        self.complex_bn = ComplexBatchNorm1d(complex_ch)

        # Mobile units + channel attention
        self.mobile_units = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_ch = real_ch
        for _ in range(n_mobileunit):
            out_ch = real_ch  # stays 32
            self.mobile_units.append(DWConvMobile(in_ch, out_ch, kernel_size))
            self.attentions.append(ChannelAttention(out_ch))
            in_ch = out_ch

        self.n_mobileunit = n_mobileunit
        self.classifier = nn.Linear(real_ch, num_classes)

    def forward(self, x):
        # x: (B, 2, 128)  →  treat as (B, 2, L) with 2 real channels = 1 complex channel
        # ComplexConv1d expects (B, 2*C_in, L) where C_in=1
        x = self.complex_conv(x)
        x = self.complex_bn(x)
        x = F.relu(x, inplace=True)

        features = []
        for i in range(self.n_mobileunit):
            x = self.mobile_units[i](x)
            x = self.attentions[i](x)
            if i >= 3:  # collect last 3 stages
                features.append(x.mean(dim=2))  # GAP

        # Feature fusion: add last 3 GAP features
        f = features[0]
        for feat in features[1:]:
            f = f + feat

        return self.classifier(f)


def build_ulcnn_torch_model(input_shape, num_classes):
    return ULCNN(num_classes=num_classes, n_neuron=16, n_mobileunit=6, kernel_size=5)
