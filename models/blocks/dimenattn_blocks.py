import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze and Excitation block"""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


class STABlock(nn.Module):
    """Spectral-Temporal Attention Block"""

    def __init__(
        self,
        channels: int,
        freq_bins: int,
        time_bins: int,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.freq_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.time_conv = nn.Conv1d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.freq_conv.bias)
        nn.init.zeros_(self.time_conv.bias)

        self.gamma = nn.Parameter(torch.tensor(init_scale))

        self.freq_bins = freq_bins
        self.time_bins = time_bins

    def _spectral_attn(self, x):
        avgF = x.mean(dim=3)
        maxF, _ = x.max(dim=3)
        y = torch.cat([avgF, maxF], dim=2)
        y = self.freq_conv(y)
        y1, y2 = y.chunk(2, dim=2)
        y = (y1 + y2) / 2
        attnF = F.softmax(y, dim=2).unsqueeze(-1)
        return attnF

    def _temporal_attn(self, x):
        avgT = x.mean(dim=2)
        maxT, _ = x.max(dim=2)
        y = torch.cat([avgT, maxT], dim=2)
        y = self.time_conv(y)
        y1, y2 = y.chunk(2, dim=2)
        y = (y1 + y2) / 2
        attnT = F.softmax(y, dim=2).unsqueeze(2)
        return attnT

    def forward(self, x):
        aF = self._spectral_attn(x)
        aT = self._temporal_attn(x)
        attn = aF * aT * self.gamma
        return x * attn
