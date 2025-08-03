import random

from torch.utils.data import Dataset

from .transforms import *


class MelSpecDataset(Dataset):
    def __init__(
        self,
        file_paths,
        bg_noise_list=None,
        rir_noise_list=None,
        split="train",
        segment_seconds=10,
        sample_rate=40000,
        num_queries=3,
        window_size=2560,
        overlap_ratio=0.5,
        fs=40000,
        n_mels=256,
    ):
        self.file_paths = file_paths
        self.bg_noise_list = bg_noise_list
        self.rir_noise_list = rir_noise_list
        self.split = split
        self.sample_rate = sample_rate
        self.num_queries = num_queries
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.fs = fs
        self.n_mels = n_mels
        self.segment_samples = int(segment_seconds * sample_rate)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_paths[idx]
        channels, _ = read_audio(file)
        waveform = channels[0]
        total_len = waveform.shape[0]

        doc_wave = waveform
        doc_spec = audio_to_melspec(
            doc_wave,
            self.window_size,
            self.overlap_ratio,
            self.fs,
            self.n_mels,
        )

        queries = []
        if self.split == "train":
            for _ in range(self.num_queries):
                if total_len > self.segment_samples:
                    start = random.randint(0, total_len - self.segment_samples)
                else:
                    start = 0
                end = start + self.segment_samples
                query_wave = waveform[start:end]
                query_wave_aug = augmentation_pipeline(
                    query_wave, self.bg_noise_list, self.rir_noise_list
                )
                query_spec = audio_to_melspec(
                    query_wave_aug,
                    self.window_size,
                    self.overlap_ratio,
                    self.fs,
                    self.n_mels,
                )
                queries.append(query_spec)
        elif self.split in ["val", "test"]:
            start = min(
                int(10 * self.sample_rate), max(0, total_len - self.segment_samples)
            )
            end = start + self.segment_samples
            query_wave = waveform[start:end]
            query_spec = audio_to_melspec(query_wave)
            queries.append(query_spec)
        else:
            raise ValueError("Wrong split keyword. Expected 'train', 'val' or 'test'.")
        return queries, doc_spec
