import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Skip connection
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial CNN layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Residual blocks
        self.block1 = ResidualBlock(32, 32, stride=(1, 1))
        self.block2 = ResidualBlock(32, 64, stride=(2, 2))
        self.block3 = ResidualBlock(64, 128, stride=(2, 2))
        self.block4 = ResidualBlock(128, 256, stride=(2, 2))
        self.block5 = ResidualBlock(256, 512, stride=(2, 2))
        self.block6 = ResidualBlock(512, 1024, stride=(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        # x shape: (B, 1, 256, T)
        x = self.conv0(x)  # -> (B,32,256,T)
        x = self.block1(x)  # -> (B,32,256,T)
        x = self.block2(x)  # -> (B,64,128,T/2)
        x = self.block3(x)  # -> (B,128,64,T/4)
        x = self.block4(x)  # -> (B,256,32,T/8)
        x = self.block5(x)  # -> (B,512,16,T/16)
        x = self.block6(x)  # -> (B,1024,8,T/32)
        x = self.adaptive_pool(x)  # -> (B,1024,3,3)
        return x


class SpectralTemporalAttention(nn.Module):
    def __init__(self, channels, freq_bins, time_bins, scale):
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.scale = scale

        # Frequency Attention Weight (C, F, 1)
        self.spectral_weight = nn.Parameter(torch.randn(1, channels, freq_bins))
        # Time Attention Weight (C, T, 1)
        self.temporal_weight = nn.Parameter(torch.randn(1, channels, time_bins))

    def forward(self, x):
        # X shape: (B, C, F, T)
        # B, C, F, T = x.shape

        # --------------------------
        # Spectral Attention (over F)
        # --------------------------
        spectral_avg = x.mean(dim=3)  # (B, C, F)
        weighted_spectral = spectral_avg * self.spectral_weight  # (B, C, F)
        spectral_attention = F.softmax(weighted_spectral, dim=2)  # (B, C, F)
        spectral_attention = spectral_attention.unsqueeze(-1)  # (B, C, F, 1)

        # --------------------------
        # Temporal Attention (over T)
        # --------------------------
        temporal_avg = x.mean(dim=2)  # (B, C, T)
        weighted_temporal = temporal_avg * self.temporal_weight  # (B, C, T)
        temporal_attention = F.softmax(weighted_temporal, dim=2)  # (B, C, T)
        temporal_attention = temporal_attention.unsqueeze(2)  # (B, C, 1, T)

        # --------------------------
        # Combine Attentions
        # --------------------------
        attention_mask = spectral_attention * temporal_attention  # (B, C, F, T)
        attention_mask = attention_mask * self.scale

        x = x * attention_mask  # (B, C, F, T)
        x = x.view(x.size(0), -1)  # flatten to (B, 9216)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Split in_dim into embedding_dim branches of length branch_dim
        branch_dim = in_dim // embedding_dim
        # First linear layer for each branch (implemented as 1x1 conv with groups)
        self.fc1 = nn.Conv1d(
            branch_dim * embedding_dim,
            hidden_dim * embedding_dim,
            kernel_size=1,
            groups=embedding_dim,
        )
        # Second linear layer for each branch
        self.fc2 = nn.Conv1d(
            hidden_dim * embedding_dim,
            1 * embedding_dim,
            kernel_size=1,
            groups=embedding_dim,
        )
        self.act = nn.ELU()

    def forward(self, x):
        # x shape: (B, in_dim)
        B = x.size(0)
        # Reshape to (B, embedding_dim, branch_dim) then view as (B, branch_dim*embedding_dim, 1) for conv1d
        branch_dim = x.shape[1] // self.embedding_dim
        x = x.view(B, self.embedding_dim * branch_dim, 1)
        out = self.fc1(x)  # (B, hidden_dim * embedding_dim, 1)
        out = self.act(out)
        out = self.fc2(out)  # (B, 1 * embedding_dim, 1)
        out = out.view(B, self.embedding_dim)  # (B, embedding_dim)
        # L2 normalize each embedding vector
        out = out / out.norm(p=2, dim=1, keepdim=True)
        return out


class DimensionMaskedResNet(nn.Module):
    def __init__(
        self,
        channels=1024,
        freq_bins=3,
        time_bins=3,
        scale=100,
        in_dim=9216,
        embedding_dim=128,
        hidden_dim=32,
    ):
        super().__init__()
        self.encoder = AudioEncoder()
        self.attention = SpectralTemporalAttention(
            channels, freq_bins, time_bins, scale
        )
        self.proj = ProjectionHead(in_dim, embedding_dim, hidden_dim)

    def forward(self, mel_spec):
        features = self.encoder(mel_spec)  # (B, C=1024, F=3, T=3)
        features_att = self.attention(
            features
        )  # apply attention and flatten to (B, 9216)
        emb = self.proj(features_att)  # (B, 128) normalized
        return emb
