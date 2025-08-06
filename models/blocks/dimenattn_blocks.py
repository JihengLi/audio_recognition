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
        F: int,
        T: int,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.F, self.T = F, T
        self.freq_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.time_conv = nn.Conv1d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.freq_conv.bias)
        nn.init.zeros_(self.time_conv.bias)
        self.gamma = nn.Parameter(torch.full([], init_scale))

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
        assert x.size(2) == self.F and x.size(3) == self.T
        aF = self._spectral_attn(x)
        aT = self._temporal_attn(x)
        attn = aF * aT * self.gamma
        return x * attn


class STASimpleBlock(nn.Module):
    """Spectral-Temporal Attention Block"""

    def __init__(
        self,
        channels: int,
        F: int,
        T: int,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.F, self.T = F, T
        self.init_scale = init_scale

        self.spectral_weight = nn.Parameter(torch.randn(1, channels, F))
        self.temporal_weight = nn.Parameter(torch.randn(1, channels, T))

    def _spectral_attn(self, x: torch.Tensor) -> torch.Tensor:
        spectral_avg = x.mean(dim=3)
        weighted = spectral_avg * self.spectral_weight
        attnF = F.softmax(weighted, dim=2).unsqueeze(-1)
        return attnF

    def _temporal_attn(self, x: torch.Tensor) -> torch.Tensor:
        temporal_avg = x.mean(dim=2)
        weighted = temporal_avg * self.temporal_weight
        attnT = F.softmax(weighted, dim=2).unsqueeze(2)
        return attnT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == self.F and x.size(3) == self.T
        aF = self._spectral_attn(x)
        aT = self._temporal_attn(x)
        attn = aF * aT * self.init_scale
        return x * attn


class STA2Block(nn.Module):
    """Spectral-Temporal Attention Block with Bottleneck mechanism"""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        init_scale: float = 5.0,
    ):
        super().__init__()
        self.freq_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.time_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.full([], init_scale))

    @staticmethod
    def _pool_sum(x, dim: int):
        return x.mean(dim=dim) + x.amax(dim=dim)

    def _freq_attn(self, x):
        y = self._pool_sum(x, dim=3)
        y = self.freq_mlp(y)
        y = torch.sigmoid(y).unsqueeze(-1)
        return y

    def _time_attn(self, x):
        y = self._pool_sum(x, dim=2)
        y = self.time_mlp(y)
        y = torch.sigmoid(y).unsqueeze(2)
        return y

    def forward(self, x):
        aF = self._freq_attn(x)
        aT = self._time_attn(x)
        attn = aF * aT * self.gamma
        return x * attn
