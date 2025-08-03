import random
import torchvision

from torch.utils.data import Dataset
from .transforms import *


class STFTDataset(Dataset):
    """Waveform -> STFT magnitude (fmin-fmax) resized to 224×224×3."""

    def __init__(
        self,
        file_paths: Sequence[str],
        *,
        split: str = "train",
        segment_sec: int = 10,
        sample_rate: int = 40_000,
        window_size: int = 2_560,
        overlap_ratio: float = 0.5,
        f_min: float = 50,
        f_max: float = 350,
        augment: Optional[WaveformAugment] = None,
        num_queries: int = 3,
        normalize: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        self.paths = list(file_paths)
        self.split = split
        self.seg_len = int(segment_sec * sample_rate)
        self.sr = sample_rate
        self.num_queries = num_queries
        self.device = device or torch.device("cpu")

        self.stft_cfg = dict(
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
            device=self.device,
        )
        self.augment = augment.to(device) if split == "train" else None
        self.resize = torchvision.transforms.Resize((224, 224))

    def __len__(self):
        return len(self.paths)

    def _rand_crop(self, wav: torch.Tensor):
        total = wav.shape[-1]
        if total <= self.seg_len:
            return wav
        if self.split == "train":
            start = random.randint(0, total - self.seg_len)
        else:
            start = max(0, (total - self.seg_len) // 2)
        return wav[start : start + self.seg_len]

    def _spec_3ch(self, wav: torch.Tensor):
        spec = audio_to_stft(wav, **self.stft_cfg)
        img = self.resize(spec.repeat(3, 1, 1))  # (3,H,W)
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav, _ = read_audio(path, target_fs=self.sr, mono=True)
        wav_full = wav[0].to(self.device, non_blocking=True)
        doc_img = self._spec_3ch(wav_full)
        queries = []
        qn = self.num_queries if self.split == "train" else 1
        for _ in range(qn):
            seg = self._rand_crop(wav_full)
            if self.augment:
                seg = self.augment(seg.to(self.device))
            queries.append(self._spec_3ch(seg))
        return queries, doc_img
