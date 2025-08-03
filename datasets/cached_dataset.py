import torch, os

from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, cache_file: str):
        self.data = torch.load(cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyCachedDataset(Dataset):
    def __init__(self, cache_dir: str):
        meta_path = os.path.join(cache_dir, "meta.pt")
        self.paths = torch.load(meta_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = torch.load(self.paths[idx])
        return sample
