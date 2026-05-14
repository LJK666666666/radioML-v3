"""PyTorch FEA-T adapter that supports both 128 and 1024 sequence lengths."""

import torch
import torch.nn as nn


def _trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


class MHSA_Block(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=16,
        d_fix_v=16,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        attn_res=False,
    ):
        super().__init__()
        self.df_qk = d_fix_qk
        self.df_v = d_fix_v
        self.h_qk = n_head_qk
        self.h_v = n_head_v
        self.attn_res = attn_res
        self.scale = float(d_fix_qk) ** 0.5

        self.to_q = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_k = nn.Linear(d_model, d_fix_qk * n_head_qk, bias=bias)
        self.to_v = nn.Linear(d_model, d_fix_v * n_head_v, bias=bias)
        self.proj_bf = nn.Conv2d(n_head_qk, n_head_qk, (1, 1), bias=False) if talking else nn.Identity()
        self.proj_v = nn.Linear(d_fix_v * n_head_v, d_model, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dp_attn = nn.Dropout(dropout)
        self.dp_v = nn.Dropout(dropout)

    def forward(self, x, attn_=None):
        bsz, n_patch, _ = x.shape
        attn_residual = attn_ if attn_ is not None else 0.0

        q = self.to_q(x).reshape(bsz, n_patch, self.h_qk, self.df_qk).permute(0, 2, 1, 3) / self.scale
        k = self.to_k(x).reshape(bsz, n_patch, self.h_qk, self.df_qk).permute(0, 2, 1, 3)
        v = self.to_v(x).reshape(bsz, n_patch, self.h_v, self.df_v).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.proj_bf(attn)
        attn = self.softmax(attn + attn_residual)
        attn = self.dp_attn(attn)
        x = (attn @ v).transpose(1, 2).reshape(bsz, n_patch, self.df_v * self.h_v)
        x = self.dp_v(self.proj_v(x))

        if self.attn_res:
            return x, attn
        return x, None


class DB_GLU_Block(nn.Module):
    def __init__(self, d_model=64, dim_feedforward=128, dropout=0.0, activate="gelu", merge_method="add"):
        super().__init__()
        self.merge_method = merge_method
        self.dim_f = dim_feedforward // 2
        dim_linear2 = dim_feedforward // 2 if merge_method != "cat" else dim_feedforward
        if merge_method == "cross_add" and dropout == 0.0:
            dropout = 0.1
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout) if merge_method == "cross_add" else nn.Identity()
        self.linear2 = nn.Linear(dim_linear2, d_model)

        if activate == "relu":
            self.act = nn.ReLU()
        elif activate == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.GELU()

    def forward(self, x):
        x_ = self.linear1(x)
        x_1, x_2 = x_[..., : self.dim_f], x_[..., self.dim_f :]
        if self.merge_method == "add":
            return self.linear2(self.dropout(x_1 * self.act(x_2) + x_2 * self.act(x_1)))
        if self.merge_method == "cross_add":
            return self.linear2(self.dropout(x_1 * self.act(x_2)) + self.dropout_2(x_2 * self.act(x_1)))
        return self.linear2(self.dropout(torch.cat((x_1 * self.act(x_2), x_2 * self.act(x_1)), dim=-1)))


class Frame(nn.Module):
    def __init__(self, patch_size=32, overlap=0.5):
        super().__init__()
        self.stride = max(1, int(patch_size * overlap))
        self.patch_size = patch_size
        self.embedding = nn.Linear(patch_size * 2, patch_size * 2)

    def forward(self, x):
        # x: [B, L, 2]
        x_i = x[:, :, 0].unfold(1, self.patch_size, self.stride)
        x_q = x[:, :, 1].unfold(1, self.patch_size, self.stride)
        feat = torch.cat((x_i, x_q), dim=-1)
        return self.embedding(feat)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=64,
        d_fix_qk=64,
        d_fix_v=16,
        d_mid=256,
        n_head_qk=4,
        n_head_v=4,
        dropout=0.0,
        bias=False,
        talking=True,
        real_former=True,
        activation="gelu",
        ffn_type="DB_GLU_add",
    ):
        super().__init__()
        self.attn_res = real_former
        self.mhsa = MHSA_Block(
            d_model=d_model,
            d_fix_qk=d_fix_qk,
            d_fix_v=d_fix_v,
            n_head_qk=n_head_qk,
            n_head_v=n_head_v,
            dropout=dropout,
            bias=bias,
            talking=talking,
            attn_res=real_former,
        )

        merge_method = "add"
        if ffn_type.startswith("DB_GLU_"):
            merge_method = ffn_type[7:]
        self.ffn = DB_GLU_Block(
            d_model=d_model,
            dim_feedforward=d_mid,
            dropout=dropout,
            activate=activation,
            merge_method=merge_method,
        )
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, attn_=None):
        x_sa, attn_ = self.mhsa(x, attn_)
        x = self.norm_1(x + x_sa)
        x = self.norm_2(x + self.ffn(x))
        return x, attn_


class FEA_T(nn.Module):
    def __init__(
        self,
        patch_size=16,
        d_fix_qk=16,
        d_fix_v=16,
        seq_length=128,
        hidden_features=64 * 4,
        n_head_qk=4,
        n_head_v=4,
        overlap=0.2,
        dropout=0.0,
        layer_num=8,
        num_class=11,
        pos_emb=True,
        bias=False,
        talking=False,
        real_former=False,
        ffn_type="DB_GLU_add",
        activation="gelu",
    ):
        super().__init__()
        self.if_pos_emb = pos_emb
        stride = max(1, int(patch_size * overlap))
        n_patch = int((seq_length - patch_size) / stride + 2)
        in_features = patch_size * 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_features))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patch, in_features))
        self.embedding = Frame(patch_size=patch_size, overlap=overlap)

        self.enc = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=in_features,
                    d_fix_qk=d_fix_qk,
                    d_fix_v=d_fix_v,
                    d_mid=hidden_features,
                    n_head_qk=n_head_qk,
                    n_head_v=n_head_v,
                    dropout=dropout,
                    bias=bias,
                    talking=talking,
                    real_former=real_former,
                    activation=activation,
                    ffn_type=ffn_type,
                )
                for _ in range(layer_num)
            ]
        )

        self.classifier = nn.Linear(in_features, num_class)
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _normalize_input(x):
        # supports [B,2,L], [B,L,2], [B,1,L,2]
        if x.ndim == 4 and x.shape[1] == 1 and x.shape[-1] == 2:
            return x[:, 0, :, :]
        if x.ndim == 3 and x.shape[-1] == 2:
            return x
        if x.ndim == 3 and x.shape[1] == 2:
            return x.transpose(1, 2)
        raise ValueError(f"Unsupported FEA_T input shape: {tuple(x.shape)}")

    def forward(self, x, state=None):
        x = self._normalize_input(x)
        bsz = x.shape[0]
        dl_y = self.embedding(x)
        cls_token = self.cls_token.expand(bsz, -1, -1)
        dl_y = torch.cat((dl_y, cls_token), dim=1)
        if self.if_pos_emb:
            dl_y = dl_y + self.pos_embed

        attn_res = None
        for layer in self.enc:
            dl_y, attn_res = layer(dl_y, attn_res)
        return self.classifier(dl_y[:, -1])


def _infer_seq_len(input_shape):
    if len(input_shape) == 2:
        if input_shape[0] == 2:
            return int(input_shape[1])
        if input_shape[1] == 2:
            return int(input_shape[0])
    if len(input_shape) == 3 and input_shape[0] == 1 and input_shape[2] == 2:
        return int(input_shape[1])
    raise ValueError(f"Unable to infer sequence length from input_shape={input_shape}")


def build_fea_t_model(input_shape, num_classes):
    seq_len = _infer_seq_len(input_shape)
    if seq_len == 128:
        return FEA_T(
            patch_size=16,
            seq_length=128,
            overlap=0.2,
            num_class=num_classes,
            hidden_features=64 * 4,
        )
    if seq_len == 1024:
        return FEA_T(
            patch_size=32,
            seq_length=1024,
            overlap=0.5,
            num_class=num_classes,
            hidden_features=64 * 4,
        )
    raise ValueError(f"FEA_T supports sequence lengths 128 or 1024, got {seq_len}")
