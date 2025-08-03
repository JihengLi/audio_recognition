import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from models import *
from datasets import *
from utils import *


class ResNet18Config:
    def __init__(self, train_paths, val_paths, epochs, device):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = device
        self.epochs = epochs

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

    def _build_dataloaders(self):
        train_dataset = CachedDataset(self.train_paths)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_dif_length,
        )

        val_dataset = CachedDataset(self.val_paths)
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn_dif_length,
        )
        return train_loader, val_loader

    def _build_model(self):
        model = ResNet18Model(embed_dim=128, hidden_size=512).to(self.device)
        model.apply(kaiming_normal_init)
        return model

    def _build_optimizer(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (no_decay if n.endswith("bias") or "bn" in n else decay).append(p)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_loader),
            eta_min=1e-06,
        )

    def _build_scaler(self):
        return GradScaler(init_scale=2**14, growth_interval=2000)
