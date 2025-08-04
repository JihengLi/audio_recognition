import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from .blocks import *
from .modules import *


class ResidualNet18(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.projection = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_features):
        features = self.resnet(input_features)
        emb = self.projection(features)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class ResidualSENet(nn.Module):
    def __init__(
        self,
        stem_cls: Type[nn.Module] = Conv7Stem,
        block_cls: Type[nn.Module] = ResidualSEBlock,
        num_in_ch: int = 1,
        layers: Tuple[int, ...] = (2, 2, 2, 2),
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        pool_shape: Tuple[int, int] = (1, 1),
        proj_hidden_dim: int = 512,
        emb_dim: int = 128,
        reduction: int = 16,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        block_kw: Dict[str, Any] = dict(reduction=reduction, drop_rate=drop_rate)
        self.encoder = ResidualEncoder(
            stem_cls=stem_cls,
            block_cls=block_cls,
            num_in_channel=num_in_ch,
            layers=layers,
            channels=channels,
            pool_shape=pool_shape,
            block_kwargs=block_kw,
        )

        C = self.encoder.out_channels
        self.emb_dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(C, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.emb_dropout(x)
        return self.fc(x)


class ResidualSTANet(nn.Module):
    def __init__(
        self,
        stem_cls: Type[nn.Module] = Conv3Stem,
        block_cls: Type[nn.Module] = ResidualBlock,
        num_in_ch: int = 1,
        layers: Tuple[int, ...] = (1, 1, 1, 1, 1, 1),
        channels: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024),
        pool_shape: Tuple[int, int] = (3, 3),
        sta_scale: float = 100.0,
        proj_hidden_dim: int = 32 * 128,
        emb_dim: int = 128,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        block_kw: Dict[str, Any] = dict(drop_rate=drop_rate)
        self.encoder = ResidualEncoder(
            stem_cls=stem_cls,
            block_cls=block_cls,
            num_in_channel=num_in_ch,
            layers=layers,
            channels=channels,
            pool_shape=pool_shape,
            block_kwargs=block_kw,
        )

        C = self.encoder.out_channels
        F, T = self.encoder.out_pool
        self.attn = STABlock(C, F, T, init_scale=sta_scale)

        self.proj = ProjectionHead(
            embedding_dim=emb_dim,
            proj_hidden_dim=proj_hidden_dim,
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.attn(feat).flatten(1)
        return self.proj(feat)
