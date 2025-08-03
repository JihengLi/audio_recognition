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
    if torch.is_tensor(channel_samples):
        y = channel_samples.detach().cpu().numpy().astype(float)
    else:
        y = np.array(channel_samples, dtype=float)
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val
    hop_length = int(window_size * overlap_ratio)
    stft_matrix = librosa.stft(
        y=y,
        n_fft=window_size,
        hop_length=hop_length,
        win_length=window_size,
        center=True,
        pad_mode="reflect",
    )
    magnitude = np.abs(stft_matrix)
    db_spec = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
        top_db=top_db,
    )
    freq_bins = db_spec.shape[0]
    freq_res = fs / window_size
    bin_min = int(f_min / freq_res)
    bin_max = int(f_max / freq_res) if f_max is not None else freq_bins
    bin_min = max(0, bin_min)
    bin_max = min(freq_bins, bin_max)
    db_spec = db_spec[bin_min:bin_max, :]
    spec_tensor = torch.tensor(db_spec, dtype=torch.float32)
    if device is not None:
        spec_tensor = spec_tensor.to(device)
    if normalize:
        mean = spec_tensor.mean(dim=1, keepdim=True)
        std = spec_tensor.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
        spec_tensor = (spec_tensor - mean) / std
    return spec_tensor


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
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
    spec = mel_transform(waveform)
    log_mel_spec = amp_to_db(spec)
    log_mel_spec = log_mel_spec.squeeze(0)
    if normalize:
        mean = log_mel_spec.mean(dim=1, keepdim=True)
        std = log_mel_spec.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)
        log_mel_spec = (log_mel_spec - mean) / std
    return log_mel_spec
