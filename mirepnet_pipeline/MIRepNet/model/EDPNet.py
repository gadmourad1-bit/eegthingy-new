
import torch
import torch.nn as nn

# from utils_1 import SSA, LightweightConv1d, Mixer1D

import torch
import torch.nn.functional as F
from torch import nn
import math
from einops import rearrange


class VarPool1D(nn.Module):

    def __init__(self, kernel_size: int, stride: int = None, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.eps = eps

    def forward(self, x):  # x: [B, C, T]
        # unfold → [B, C, kernel, L]
        patches = x.unfold(dimension=-1,
                           size=self.kernel_size,
                           step=self.stride)
        # var over kernel dim, keep L
        var = patches.var(dim=-2) + self.eps  # [B, C, L]
        return var


class VarMaxPool1D(nn.Module):
    def __init__(self, total_T: int, kernel_size: int):
        super().__init__()
        self.var_pool = VarPool1D(kernel_size=kernel_size,
                                  stride=kernel_size)

    def forward(self, x):                      # x: [B,C,T]
        var = self.var_pool(x)                 # [B,C,L]
        return var.max(dim=-1, keepdim=True).values   # → [B,C,1]


class LightweightConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=1,
        depth_multiplier=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        # input = input.view(-1, H, T)
        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=H)  
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=B)  #  (B*C/H, H, T) -> (B, C, T)

        return output


class SSA(nn.Module):

    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(T, 250)   # torch.Size([16, 531, 500])  -->  torch.Size([16, 531, 1])

    def forward(self, x):
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])
            out.append(x)
        y = torch.concat(out, dim=1)
        return y

class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )
        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):
        x = self.time_conv(x)    # LightweightConv1d
        x, _ = self.ssa(x)
        x_chan = self.chanConv(x)
        feature = self.mixer(x_chan)

        return feature


class EDPNet(nn.Module):

    def __init__(
        self,
        chans,
        samples,
        num_classes=3,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=[50, 100, 200],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]

        # *Inter-class Separation Prototype(ISP)
        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        # *Intra-class Compactness(ICP)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x):
        features = self.encoder(x)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)

        # return features, logits, self.isp
        return logits


class PrototypeLoss(nn.Module):

    def forward(self, features, proxy, labels):

        label_prototypes = torch.index_select(proxy, dim=0, index=labels)

        pl = huber_loss(features, label_prototypes, sigma=1)
        pl_loss = torch.mean(pl)

        return pl_loss


def huber_loss(input, target, sigma=1):
    beta = 1.0 / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)


    return torch.sum(loss, dim=1)
