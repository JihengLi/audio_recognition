import torch
import torch.nn as nn

from typing import Optional


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        proj_hidden_dim: int = 32 * 128,
        l2_norm: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert proj_hidden_dim % self.embedding_dim == 0
        self.hidden_dim = proj_hidden_dim // self.embedding_dim
        self.l2_norm = l2_norm
        self.fc1: Optional[nn.Conv1d] = None
        self.fc2: Optional[nn.Conv1d] = None
        self.act = nn.ELU()

    def _build(self, in_dim: int, device):
        assert in_dim % self.embedding_dim == 0
        branch = in_dim // self.embedding_dim
        self.fc1 = nn.Conv1d(
            in_channels=branch * self.embedding_dim,
            out_channels=self.hidden_dim * self.embedding_dim,
            kernel_size=1,
            groups=self.embedding_dim,
        ).to(device)
        self.fc2 = nn.Conv1d(
            in_channels=self.hidden_dim * self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            groups=self.embedding_dim,
        ).to(device)

    def forward(self, x):
        B, D = x.shape
        if self.fc1 is None:
            self._build(D, x.device)

        branch = D // self.embedding_dim
        x = x.view(B, self.embedding_dim * branch, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x).view(B, self.embedding_dim)

        if self.l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        return x
