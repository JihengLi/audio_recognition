import random, torch
import torch.nn.functional as F

from .audios import read_audio


def match_length(noise, target_length):
    current_length = noise.shape[0]
    if current_length == target_length:
        return noise
    elif current_length > target_length:
        return noise[:target_length]
    else:
        repeats = (target_length // current_length) + 1
        noise_extended = noise.repeat(repeats)[:target_length]
        return noise_extended


def augmentation_pipeline(
    waveform,
    bg_noise_list=None,
    rir_noise_list=None,
    p_gain=0.5,
    p_noise=0.5,
    p_reverb=0.5,
    p_time_stretch=0,
    p_pitch_shift=0.5,
    p_time_shift=0.5,
    gain_db_range=(-6, 6),
    snr_range=(5, 25),
    time_stretch_range=(0.8, 1.25),
    pitch_shift_semitone_range=(-4, 4),
    max_time_shift_fraction=0.1,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(waveform):
        waveform = torch.tensor(waveform, dtype=torch.float32).to(device)
    augmented = waveform.clone()

    # 1. Stoachastic gain
    if random.random() < p_gain:
        gain_db = random.uniform(gain_db_range[0], gain_db_range[1])
        gain_factor = 10 ** (gain_db / 20)
        augmented = augmented * gain_factor

    # 2. Stochastic noise injection
    if (bg_noise_list is not None) and (random.random() < p_noise):
        bg_noise_file = random.choice(bg_noise_list)
        noise_channels, _ = read_audio(bg_noise_file)
        noise_waveform = noise_channels[0]
        if not torch.is_tensor(noise_waveform):
            noise_waveform = torch.tensor(noise_waveform, dtype=torch.float32).to(
                device
            )
        noise_waveform = match_length(noise_waveform, augmented.shape[0])
        snr_db = random.uniform(snr_range[0], snr_range[1])
        sig_power = augmented.pow(2).mean()
        noise_power = noise_waveform.pow(2).mean()
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        scaling_factor = (target_noise_power / noise_power).sqrt()
        scaled_noise = noise_waveform * scaling_factor
        augmented = augmented + scaled_noise

    # 3. Room Impulse Response (RIR) reverb
    if (rir_noise_list is not None) and (random.random() < p_reverb):
        rir_file = random.choice(rir_noise_list)
        rir_channels, _ = read_audio(rir_file)
        rir_waveform = rir_channels[0]
        if not torch.is_tensor(rir_waveform):
            rir_waveform = torch.tensor(rir_waveform, dtype=torch.float32).to(device)
        rir_waveform = rir_waveform / rir_waveform.abs().max()
        augmented_ = augmented.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        rir = rir_waveform.unsqueeze(0).unsqueeze(0)
        pad = rir.shape[-1] // 2
        augmented_ = F.conv1d(augmented_, rir, padding=pad)
        augmented = augmented_.squeeze(0).squeeze(0)

    # 4. Time stretch (simulate speed change)
    if random.random() < p_time_stretch:
        stretch_factor = random.uniform(time_stretch_range[0], time_stretch_range[1])
        orig_length = augmented.shape[-1]
        new_length = int(orig_length / stretch_factor)
        augmented_ = F.interpolate(
            augmented.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        augmented = F.interpolate(
            augmented_, size=orig_length, mode="linear", align_corners=False
        ).squeeze()

    # 5. Pitch shift (simulate pitch change)
    if random.random() < p_pitch_shift:
        n_semitones = random.uniform(
            pitch_shift_semitone_range[0], pitch_shift_semitone_range[1]
        )
        factor = 2 ** (n_semitones / 12)
        orig_length = augmented.shape[-1]
        new_length = int(orig_length / factor)
        augmented_ = F.interpolate(
            augmented.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        augmented = F.interpolate(
            augmented_, size=orig_length, mode="linear", align_corners=False
        ).squeeze()

    # 6. Stochastic time shifting
    if random.random() < p_time_shift:
        max_shift = int(max_time_shift_fraction * augmented.shape[-1])
        shift = random.randint(-max_shift, max_shift)
        if shift > 0:
            augmented = torch.cat(
                (augmented[shift:], torch.zeros(shift, device=augmented.device)), dim=0
            )
        elif shift < 0:
            augmented = torch.cat(
                (torch.zeros(-shift, device=augmented.device), augmented[:shift]), dim=0
            )

    return augmented.cpu().numpy()
