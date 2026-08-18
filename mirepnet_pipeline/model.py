# Vendored from staraink/MIRepNet (model/mlm.py), MIT licensed.
# Trimmed: dropped the `wandb` import (only used by the original repo's own
# training script, not by the model itself) and the pretraining-only decoder
# heads we don't need for downstream fine-tuning/inference.
#
# Architecture (unchanged from upstream):
#   PatchEmbedding: (B, C, T) -> 1D temporal conv -> spatial conv across all
#     C channels -> BN/ELU/AvgPool -> linear projection to emb_size.
#   TransformerEncoder: `depth` standard pre-norm transformer blocks over the
#     resulting token sequence.
#   clshead: linear classification head (downstream mode).
#
# Pretrained checkpoint (weight/MIRepNet.pth) has emb_size=256, depth=6,
# num_channels=45 baked into its tensor shapes - keep these defaults unless
# you are pretraining from scratch or using a custom channel count.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=128, num_channels=45):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 25), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(self.num_channels, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        self.projection = nn.Sequential(
            nn.Conv2d(128, embed_dim, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # NOTE: chan_embed is kept only for checkpoint compatibility with the
        # original 45-channel pretrained model. It is not used in forward().
        self.chan_embed = nn.Embedding(45, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        q = self.queries(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.keys(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.values(x).view(B, N, self.num_heads, -1).transpose(1, 2)
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        att = F.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, dropout=0.5):
        super().__init__(*[TransformerEncoderBlock(emb_size, drop_p=dropout) for _ in range(depth)])


class MIRepNet(nn.Module):
    """Downstream-mode MIRepNet: embedding -> transformer -> linear head.

    forward(x) with x: (B, C, T) -> (pooled: (B, emb_size), logits: (B, n_classes))
    where C can be 45 (original template) or your native channel count (e.g. 15).
    """

    def __init__(self, emb_size=256, depth=6, n_classes=2, num_channels=45, pretrain_path=None):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = PatchEmbedding(embed_dim=emb_size, num_channels=num_channels)
        self.transformer = TransformerEncoder(depth, emb_size, dropout=0.5)
        self.clshead = nn.Linear(emb_size, n_classes)
        if pretrain_path is not None:
            self.load_pretrained(pretrain_path)

    def forward(self, x: torch.Tensor):
        feats = self.embedding(x)
        feats = self.transformer(feats)
        pooled = feats.mean(dim=1)
        logits = self.clshead(pooled)
        return pooled, logits

    def load_pretrained(self, path, freeze_encoder=False, strict=False):
        """Load the released MIRepNet.pth backbone or a fine-tuned checkpoint.

        The classification head (clshead) is included only if present in the
        checkpoint (e.g. your fine-tuned file).

        If the pretrained checkpoint uses 45 channels but this model uses a
        different number (e.g. 15), only the temporal conv, transformer, and
        projection layers will be loaded. The spatial conv (conv2) will be
        re-initialized randomly for the new channel count.
        """
        state = torch.load(path, map_location='cpu')
        own = self.state_dict()
        matched = {}
        skipped = []

        for k, v in state.items():
            if k in own and v.shape == own[k].shape:
                matched[k] = v
            elif k in own:
                skipped.append(k)

        own.update(matched)
        self.load_state_dict(own, strict=False)

        if freeze_encoder:
            for name, p in self.named_parameters():
                if 'embedding' in name or 'transformer' in name:
                    p.requires_grad = False

        print(f"[MIRepNet] loaded {len(matched)}/{len(own)} tensors from {path}")
        if skipped:
            print(f"[MIRepNet] re-initialized {len(skipped)} tensors (shape mismatch):")
            for k in skipped:
                print(f"  - {k}: checkpoint={state[k].shape}, model={own[k].shape}")