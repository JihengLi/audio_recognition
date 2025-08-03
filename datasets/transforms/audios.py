import numpy as np
import torch, torchaudio, librosa

from typing import Union, List, Optional

import torchaudio
import torch
from typing import Tuple


def read_audio(
    file_path: str,
    target_fs: int = 40000,
    mono: bool = True,
    normalize: bool = False,
) -> Tuple[torch.Tensor, int]:
    waveform, orig_fs = torchaudio.load(file_path)
    if orig_fs != target_fs:
        resampler = torchaudio.transforms.Resample(orig_fs, target_fs)
        waveform = resampler(waveform)
        fs = target_fs
    else:
        fs = orig_fs
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if normalize:
        max_val = waveform.abs().max().clamp(min=1e-5)
        waveform = waveform / max_val
    return waveform, fs


def audio_to_stft(
    channel_samples: Union[List[float], torch.Tensor],
    window_size: int,
    overlap_ratio: float,
    fs: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    top_db: float = 80.0,
    normalize: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or (
        channel_samples.device if torch.is_tensor(channel_samples) else "cpu"
    )
    if not torch.is_tensor(channel_samples):
        y = torch.tensor(channel_samples, dtype=torch.float32, device=device)
    else:
        y = channel_samples.to(device, dtype=torch.float32)
    max_val = y.abs().max()
    if max_val > 1.0:
        y = y / max_val
    hop = int(window_size * overlap_ratio)
    win = torch.hann_window(window_size, device=device)
    spec = torch.stft(
        y,
        n_fft=window_size,
        hop_length=hop,
        win_length=window_size,
        window=win,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    pow_spec = spec.abs().pow(2.0)
    ref = pow_spec.max()
    log_spec = 10 * torch.log10(torch.clamp(pow_spec, min=1e-10))
    log_spec = torch.clamp(log_spec, min=ref.log10() * 10 - top_db)
    freq_res = fs / window_size
    bin_min = int(f_min / freq_res)
    bin_max = int(f_max / freq_res) if f_max else pow_spec.size(0)
    log_spec = log_spec[bin_min:bin_max, :]
    if normalize:
        mu = log_spec.mean(dim=1, keepdim=True)
        sigma = log_spec.std(dim=1, unbiased=False, keepdim=True).clamp(1e-5)
        log_spec = (log_spec - mu) / sigma
    return log_spec


def audio_to_melspec(
    channel_samples: Union[List[float], torch.Tensor],
    window_size: int,
    overlap_ratio: float,
    fs: int,
    n_mels: int,
    power: float = 2.0,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    top_db: float = 80.0,
    normalize: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or (
        channel_samples.device if torch.is_tensor(channel_samples) else "cpu"
    )
    if not torch.is_tensor(channel_samples):
        waveform = torch.tensor(channel_samples, dtype=torch.float32)
    else:
        waveform = channel_samples.to(dtype=torch.float32)
    waveform = waveform.unsqueeze(0)
    if device is not None:
        waveform = waveform.to(device)
    hop_length = int(window_size * overlap_ratio)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=fs,
        n_fft=window_size,
        hop_length=hop_length,
        win_length=window_size,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        center=True,
        pad_mode="reflect",
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=top_db,
    ).to(device)
    spec = mel_transform(waveform)
    log_mel_spec = amp_to_db(spec)
    log_mel_spec = log_mel_spec
    if normalize:
        mean = log_mel_spec.mean(dim=1, keepdim=True)
        std = log_mel_spec.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
        log_mel_spec = (log_mel_spec - mean) / std
    return log_mel_spec
