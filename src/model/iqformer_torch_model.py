"""PyTorch IQFormer adapter for the unified training/evaluation pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def stemIQ(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs // 2, kernel_size=5, stride=1, padding=2, groups=in_chs),
        nn.BatchNorm1d(out_chs // 2),
    )


def stemSTFT(freq_bins, in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=(freq_bins, 1), stride=1, groups=in_chs),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
    )


class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=1, padding=1, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


class ConvEncoderIQ(nn.Module):
    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            return shortcut + self.drop_path(self.layer_scale * x)
        return shortcut + self.drop_path(x)


class FCN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttention(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = F.normalize(self.to_query(x), dim=-1)
        key = F.normalize(self.to_key(x), dim=-1)

        query_weight = query @ self.w_g
        attn = F.normalize(query_weight * self.scale_factor, dim=1)
        global_query = torch.sum(attn * query, dim=1)
        global_query = global_query.unsqueeze(1).repeat(1, key.shape[1], 1)

        out = self.proj(global_query * key) + query
        return self.final(out)


class LocalRepresentation(nn.Module):
    def __init__(self, dim, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            return shortcut + self.drop_path(self.layer_scale * x)
        return shortcut + self.drop_path(x)


class Fusion(nn.Module):
    def __init__(self, input_channel, drop):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, input_channel * 2, 1),
            nn.BatchNorm1d(input_channel * 2),
            nn.GELU(),
            nn.Conv1d(input_channel * 2, input_channel * 2, 1),
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, stft):
        return self.drop(self.conv(torch.cat((x, stft), dim=1)))


class IQFormerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.local_representation = LocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0.0, use_layer_scale=True
        )
        self.attn = EfficientAdditiveAttention(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = FCN(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True
            )

    def forward(self, x):
        x = self.local_representation(x)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
            return x
        x = x + self.drop_path(self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
        x = x + self.drop_path(self.linear(x))
        return x


def stage(
    dim,
    index,
    layers,
    mlp_ratio=4.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    vit_num=1,
):
    blocks = []
    total_blocks = max(1, sum(layers) - 1)
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / total_blocks
        if layers[index] - block_idx <= vit_num:
            blocks.append(
                IQFormerEncoder(
                    dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            blocks.append(ConvEncoderIQ(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))
    return nn.Sequential(*blocks)


def _normalize_iq_input(x):
    if x.ndim != 3:
        raise ValueError(f"IQFormer expects 3D tensor input [B,2,L] or [B,L,2], got shape {tuple(x.shape)}")
    if x.shape[1] == 2:
        return x
    if x.shape[2] == 2:
        return x.transpose(1, 2)
    raise ValueError(f"Unable to infer IQ channels from input shape {tuple(x.shape)}")


def compute_stft_features(x):
    # x: [B, 2, L]
    x = _normalize_iq_input(x)
    signal_i = x[:, 0, :]
    signal_q = x[:, 1, :]

    n_fft = 62  # onesided bins = 32
    hop_length = max(1, x.shape[-1] // 64)
    window = torch.hann_window(n_fft, device=x.device, dtype=signal_i.dtype)
    stft_i = torch.stft(
        signal_i,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
        onesided=True,
    )
    stft_q = torch.stft(
        signal_q,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
        onesided=True,
    )
    stft_mag = torch.sqrt(stft_i.abs().pow(2) + stft_q.abs().pow(2) + 1e-8)
    return stft_mag.unsqueeze(1)


class IQFormer(nn.Module):
    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=4,
        num_classes=11,
        down_patch_size=5,
        down_stride=3,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        vit_num=1,
    ):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.bn = nn.BatchNorm1d(2)
        self.bn_stft = nn.BatchNorm2d(1)
        self.patch_embed_iq = stemIQ(2, embed_dims[0] // 4)
        self.patch_embed_stft = stemSTFT(32, 1, embed_dims[0] // 4)
        self.fusion = Fusion(embed_dims[0] // 4, drop_rate)

        network = []
        for i in range(len(layers)):
            network.append(
                stage(
                    embed_dims[i],
                    i,
                    layers,
                    mlp_ratio=mlp_ratios,
                    drop_path_rate=drop_path_rate,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    vit_num=vit_num,
                )
            )
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)
        self.patch_lstm = nn.LSTM(
            input_size=embed_dims[0] // 2,
            hidden_size=embed_dims[0] // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=drop_rate,
        )
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x, stft=None):
        x = _normalize_iq_input(x)
        if stft is None:
            stft = compute_stft_features(x)

        x = self.bn(x)
        stft = self.bn_stft(stft)

        x = self.patch_embed_iq(x)
        stft = self.patch_embed_stft(stft).squeeze(2)
        if stft.shape[-1] != x.shape[-1]:
            stft = F.interpolate(stft, size=x.shape[-1], mode="linear", align_corners=False)

        x = self.fusion(x, stft)
        x, _ = self.patch_lstm(x.permute(0, 2, 1))
        x = self.forward_tokens(x.permute(0, 2, 1))
        x = self.norm(x)
        return self.head(self.global_avg_pool(x))


def build_iqformer_model(input_shape, num_classes):
    # Keep architecture aligned with IQFormer/main.py defaults for RadioML2016.
    return IQFormer(
        [2, 3, 2],
        embed_dims=[64, 64, 64],
        mlp_ratios=4,
        num_classes=num_classes,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.2,
        drop_path_rate=0.0,
        use_layer_scale=False,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        vit_num=1,
    )
