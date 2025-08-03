import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import highpass_biquad, lowpass_biquad

from .audios import read_audio


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur > target_len:
        return x[..., :target_len]
    rep = target_len // cur + 1
    return x.repeat(rep)[..., :target_len]


class WaveformAugment(nn.Module):
    def __init__(
        self,
        sample_rate: int = 40000,
        bg_noise_list: Optional[Sequence[str]] = None,
        rir_noise_list: Optional[Sequence[str]] = None,
        # probabilities
        p_gain: float = 0.5,
        p_noise: float = 0.3,
        p_reverb: float = 0.25,
        p_filter: float = 0.3,
        p_compress: float = 0.2,
        p_time_stretch: float = 0.15,
        p_pitch_shift: float = 0.4,
        p_time_shift: float = 0.5,
        # ranges
        gain_db_range: Tuple[float, float] = (-6, 6),
        snr_range: Tuple[float, float] = (10, 25),
        lowpass_freq_range: Tuple[int, int] = (3_000, 8_000),
        highpass_freq_range: Tuple[int, int] = (40, 300),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_semitone_range: Tuple[int, int] = (-3, 3),
        max_time_shift_fraction: float = 0.1,
        # compressor
        compressor_threshold: float = 0.6,
        compressor_ratio: float = 4.0,
        # reproducibility
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sr = sample_rate
        self.bg_noise_list = list(bg_noise_list or [])
        self.rir_noise_list = list(rir_noise_list or [])

        self.p_gain, self.gain_db_range = p_gain, gain_db_range
        self.p_noise, self.snr_range = p_noise, snr_range
        self.p_reverb = p_reverb
        self.p_filter = p_filter
        self.lowpass_freq_range, self.highpass_freq_range = (
            lowpass_freq_range,
            highpass_freq_range,
        )
        self.p_compress = p_compress
        self.comp_thr, self.comp_ratio = compressor_threshold, compressor_ratio
        self.p_time_stretch, self.time_stretch_range = (
            p_time_stretch,
            time_stretch_range,
        )
        self.p_pitch_shift, self.pitch_shift_semitone_range = (
            p_pitch_shift,
            pitch_shift_semitone_range,
        )
        self.p_time_shift, self.max_time_shift_fraction = (
            p_time_shift,
            max_time_shift_fraction,
        )
        self.rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def _aug_gain(self, y: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < self.p_gain:
            g = self.rng.uniform(*self.gain_db_range)
            y = y * 10 ** (g / 20)
        return y

    def _aug_noise(self, y: torch.Tensor) -> torch.Tensor:
        if not (self.bg_noise_list and self.rng.random() < self.p_noise):
            return y
        n_path = self.rng.choice(self.bg_noise_list)
        n_ch, _ = read_audio(n_path, target_fs=self.sr, mono=True)
        n = _match_length(n_ch[0].to(y.device), y.shape[-1])
        snr = self.rng.uniform(*self.snr_range)
        yp, np_ = y.pow(2).mean(), n.pow(2).mean().clamp(1e-6)
        n = n * (yp / (10 ** (snr / 10) * np_)).sqrt()
        return y + n

    def _aug_reverb(self, y: torch.Tensor) -> torch.Tensor:
        if not (self.rir_noise_list and self.rng.random() < self.p_reverb):
            return y
        r_path = self.rng.choice(self.rir_noise_list)
        r_ch, _ = read_audio(r_path, target_fs=self.sr, mono=True)
        r = r_ch[0].to(y.device)
        r /= r.abs().max()
        pad = r.shape[-1] // 2
        return F.conv1d(y[None, None], r[None, None], padding=pad).squeeze()

    def _aug_filter(self, y: torch.Tensor) -> torch.Tensor:
        if self.rng.random() >= self.p_filter:
            return y
        if self.rng.random() < 0.5:
            fc = self.rng.randint(*self.lowpass_freq_range)
            return lowpass_biquad(y, self.sr, fc)
        fc = self.rng.randint(*self.highpass_freq_range)
        return highpass_biquad(y, self.sr, fc)

    def _aug_compress(self, y: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < self.p_compress:
            m = y.abs() > self.comp_thr
            y[m] = torch.sign(y[m]) * (
                self.comp_thr + (y[m].abs() - self.comp_thr) / self.comp_ratio
            )
        return y

    def _aug_time_stretch(self, y: torch.Tensor, orig_len: int) -> torch.Tensor:
        if self.rng.random() >= self.p_time_stretch:
            return y
        sf = self.rng.uniform(*self.time_stretch_range)
        n = int(orig_len / sf)
        y = F.interpolate(y[None, None], size=n, mode="linear", align_corners=False)
        return F.interpolate(
            y, size=orig_len, mode="linear", align_corners=False
        ).squeeze()

    def _aug_pitch_shift(self, y: torch.Tensor, orig_len: int) -> torch.Tensor:
        if self.rng.random() >= self.p_pitch_shift:
            return y
        semi = self.rng.uniform(*self.pitch_shift_semitone_range)
        factor = 2 ** (semi / 12)
        n = int(orig_len / factor)
        y = F.interpolate(y[None, None], size=n, mode="linear", align_corners=False)
        return F.interpolate(
            y, size=orig_len, mode="linear", align_corners=False
        ).squeeze()

    def _aug_roll(self, y: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < self.p_time_shift:
            mx = int(self.max_time_shift_fraction * y.shape[-1])
            shift = self.rng.randint(-mx, mx)
            y = torch.roll(y, shift, -1)
        return y

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor | List[float]) -> torch.Tensor:
        device = wav.device if torch.is_tensor(wav) else "cpu"
        y = wav if torch.is_tensor(wav) else torch.tensor(wav, dtype=torch.float32)
        y = y.to(device).clone()
        orig_len = y.shape[-1]

        y = self._aug_gain(y)
        y = self._aug_noise(y)
        y = self._aug_reverb(y)
        y = self._aug_filter(y)
        y = self._aug_compress(y)
        y = self._aug_time_stretch(y, orig_len)
        y = self._aug_pitch_shift(y, orig_len)
        y = self._aug_roll(y)
        return y.clamp_(-1.0, 1.0)
