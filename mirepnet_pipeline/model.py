# Vendored from staraink/MIRepNet (model/mlm.py), MIT licensed.
# Trimmed: dropped the `wandb` import and pretraining-only decoder heads.
#
# Added ChannelAdapter + AdaptedMIRepNet to map a custom montage (e.g. 15ch)
# to the pretrained 45-channel topology via a learnable 1x1 Conv1d.

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
        # kept only for checkpoint compatibility; not used in forward
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
        Mismatched tensors (e.g. different channel count or class count) are skipped
        and remain randomly initialised.
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


class ChannelAdapter(nn.Module):
    """Maps a source montage (e.g. 15 channels) to the pretrained channel count (45)."""
    def __init__(self, in_channels=15, out_channels=45):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, in_channels, T) -> (B, out_channels, T)
        return self.conv(x)


class AdaptedMIRepNet(nn.Module):
    """Wrapper that adds a learnable channel adapter before a frozen (or partially frozen)
    MIRepNet backbone that expects 45 channels.

    Use this when the original pretrained 45-channel backbone should be kept intact.
    """
    def __init__(self, in_channels=15, n_classes=2, pretrain_path=None):
        super().__init__()
        # Create a 45-channel backbone with the original pretrained weights
        self.base_model = MIRepNet(num_channels=45, n_classes=3)  # original had 3 classes
        if pretrain_path is not None:
            self.base_model.load_pretrained(pretrain_path, strict=False)
        # Replace the classification head with the required number of classes
        self.base_model.clshead = nn.Linear(self.base_model.clshead.in_features, n_classes)
        # Add adapter
        self.adapter = ChannelAdapter(in_channels, 45)

    def forward(self, x):
        x = self.adapter(x)          # (B, 15, T) -> (B, 45, T)
        return self.base_model(x)    # returns (pooled, logits)

    def load_state_dict(self, state_dict, strict=False):
        # Custom loading to handle any potential missing keys gracefully
        own = self.state_dict()
        for k, v in state_dict.items():
            if k in own and v.shape == own[k].shape:
                own[k] = v
        super().load_state_dict(own, strict=strict)