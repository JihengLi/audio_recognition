import random
from typing import List, Sequence, Optional

import torch
from torch.utils.data import Dataset

from .transforms import *


def seed_worker(worker_id: int) -> None:
    base_seed = torch.initial_seed() % 2**32
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


class MelSpecDataset(Dataset):
    "Load waveform, apply optional augmentation, return log-mel specs."

    def __init__(
        self,
        file_paths: Sequence[str],
        split: str = "train",
        segment_sec: int = 10,
        sample_rate: int = 40_000,
        window_size: int = 2_560,
        overlap_ratio: float = 0.5,
        n_mels: int = 256,
        augment: Optional[WaveformAugment] = None,
        num_queries: int = 3,
        normalize: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}

        self.paths: List[str] = list(file_paths)
        self.split = split
        self.seg_len = int(segment_sec * sample_rate)
        self.sr = sample_rate
        self.num_queries = num_queries
        self.device = device or torch.device("cpu")

        self.mel_cfg = dict(
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            fs=sample_rate,
            n_mels=n_mels,
            normalize=normalize,
            device=self.device,
        )
        self.augment = augment.to(device) if split == "train" else None

    def __len__(self) -> int:
        return len(self.paths)

    def _rand_crop(self, wav: torch.Tensor) -> torch.Tensor:
        """Random crop for train, center crop for val/test."""
        total = wav.shape[-1]
        if total <= self.seg_len:
            return wav

        if self.split == "train":
            start = random.randint(0, total - self.seg_len)
        else:
            start = max(0, (total - self.seg_len) // 2)
        end = start + self.seg_len
        return wav[start:end]

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        wav, _ = read_audio(path, target_fs=self.sr, mono=True)
        wav_full = wav[0].to(self.device, non_blocking=True)

        doc_spec = audio_to_melspec(wav_full, **self.mel_cfg)

        queries = []
        q_n = self.num_queries if self.split == "train" else 1
        for _ in range(q_n):
            seg = self._rand_crop(wav_full)
            if self.augment:
                seg = self.augment(seg.to(self.device))
            q_spec = audio_to_melspec(seg, **self.mel_cfg)
            queries.append(q_spec)

        return queries, doc_spec
