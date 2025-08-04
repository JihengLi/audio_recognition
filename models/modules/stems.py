import torch.nn as nn


class _BaseStem(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = 1

    def forward(self, x):
        raise NotImplementedError


class Conv3Stem(_BaseStem):
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 32,
        act=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            act,
        )
        self.downsample = 1

    def forward(self, x):
        return self.net(x)


class Conv7Stem(_BaseStem):
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 32,
        act=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            act,
            nn.MaxPool2d(3, 2, 1),
        )
        self.downsample = 4  # 2 (conv) × 2 (pool)

    def forward(self, x):
        return self.net(x)


class HybridStem(_BaseStem):
    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 32,
        act=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            act,
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            act,
        )
        self.downsample = 2

    def forward(self, x):
        return self.net(x)
