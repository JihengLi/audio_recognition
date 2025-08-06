import torch

from torch.utils.data import DataLoader
from torch.amp import GradScaler

from models import *
from datasets import *
from utils import *


class ResidualSTA2NetConfig:
    def __init__(self, train_cache_path, val_cache_path, epochs, device):
        self.train_cache_path = train_cache_path
        self.val_cache_path = val_cache_path
        self.device = device
        self.epochs = epochs

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

    def _build_dataloaders(self):
        train_dataset = LazyCachedDataset(self.train_cache_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_dif_length,
        )
        val_dataset = LazyCachedDataset(self.val_cache_path)
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
        model = ResidualS2Net().to(self.device)
        dummy = make_dummy_input(model, spatial_size=(256, 256))
        _ = model(dummy)
        model.apply(kaiming_normal_init)
        return model

    def _build_optimizer(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if n.endswith("bias") or "bn" in n.lower() or n.endswith("gamma"):
                no_decay.append(p)
            else:
                decay.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-2},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=5e-4,
            betas=(0.9, 0.95),
        )
        return optimizer

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-2,
            pct_start=0.3,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    def _build_scaler(self):
        return GradScaler(init_scale=2**14, growth_interval=2000)
