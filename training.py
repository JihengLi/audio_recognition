import os, math, re
import torch

from datasets import *
from utils import *
from models import *
from losses import *
from configs import *

EPOCH_NUM = 100
NUM_AUG_QUERIES = 3
NTX_LOSS_TEM = 0.2

if __name__ == "__main__":
    train_cache_path = "data/dataset_cache/dmresnet/train"
    val_cache_path = "data/dataset_cache/dmresnet/val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DimensionMaskedResNetConfig(
        train_cache_path=train_cache_path,
        val_cache_path=val_cache_path,
        epochs=EPOCH_NUM,
        device=device,
    )
    train_loader = cfg.train_loader
    val_loader = cfg.val_loader
    model = cfg.model
    optimizer = cfg.optimizer
    scheduler = cfg.scheduler
    scaler = cfg.scaler

    start_epoch = 1
    best_val_loss = math.inf
    ckpt_dir = "outputs/checkpoints"

    latest_epoch = -1
    pat = re.compile(r"epoch(\d+)\.pth$")
    resume_path = None

    for fname in os.listdir(ckpt_dir):
        m = pat.match(fname)
        if m:
            ep = int(m.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                resume_path = os.path.join(ckpt_dir, fname)

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]

        print(
            f"Auto-resumed from {resume_path} — "
            f"epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.4f})"
        )
    else:
        print("No checkpoint found — starting fresh training.")

    train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        scaler,
        device,
        epochs=EPOCH_NUM,
        num_aug_queries=NUM_AUG_QUERIES,
        loss_fn=lambda q, d: nt_xent_loss(q, d, NTX_LOSS_TEM),
        ckpt_dir=ckpt_dir,
    )
