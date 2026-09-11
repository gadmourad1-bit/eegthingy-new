import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import argparse
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import mne
from scipy import signal
from scipy.signal import detrend

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class LogPowerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(torch.mean(x ** 2, dim=self.dim), 1e-4, 1e4))

class InterFre(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        out = sum(x_list)
        return F.gelu(out)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)

class Stem(nn.Module):
    def __init__(self, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.patch_size = patch_size

        # 第一层 1×1 Conv, groups=radix
        self.sconv = Conv(
            nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_planes),
            activation=None
        )

        self.tconv = nn.ModuleList()
        k = kernel_size
        for _ in range(self.radix):
            self.tconv.append(
                Conv(
                    nn.Conv1d(self.out_planes, self.out_planes, k, 1,
                              groups=self.out_planes, padding=k // 2, bias=False),
                    bn=nn.BatchNorm1d(self.out_planes),
                    activation=None
                )
            )
            k //= 2

        self.interFre = InterFre()
        self.power = LogPowerLayer(dim=3)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        # x: [N, C, T], C = in_planes * radix
        N, C, T = x.shape
        out = self.sconv(x)  # [N, mid_planes, T]
        out = torch.split(out, self.out_planes, dim=1)  
        out = [m(o) for o, m in zip(out, self.tconv)] 
        out = self.interFre(out)  # [N, out_planes, T]
        out = out.reshape(N, self.out_planes, T // self.patch_size, self.patch_size)
        out = self.power(out)  # [N, out_planes, T//patch_size]
        out = self.dp(out)
        return out  # [N, out_planes, T//patch_size]

class IFNet(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes):
        super().__init__()
        self.in_planes = in_planes * radix
        self.out_planes = out_planes
        self.stem = Stem(self.in_planes, self.out_planes, kernel_size,
                         patch_size=patch_size, radix=radix)

        self.fc = nn.Sequential(
            LinearWithConstraint(out_planes * (time_points // patch_size),
                                 num_classes, doWeightNorm=True),
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, C, T]
        out = self.stem(x)             # [B, out_planes, T//patch_size]
        out = self.fc(out.flatten(1))  # [B, num_classes]

        return out
