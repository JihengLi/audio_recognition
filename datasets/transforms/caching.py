import os
import torch


def atomic_save(data, cache_file):
    tmp_file = cache_file + ".tmp"
    torch.save(data, tmp_file)
    if os.path.exists(cache_file):
        os.remove(cache_file)
    os.rename(tmp_file, cache_file)


def preprocess_and_cache(
    dataset,
    cache_file: str,
    checkpoint_interval=100,
):
    if os.path.exists(cache_file):
        print(f"Cache file found. Loading dataset from '{cache_file}'...")
        try:
            data = torch.load(cache_file)
        except Exception as e:
            print(f"Error loading cache file: {e}. Starting from scratch.")
            data = []
        start_idx = len(data)
        if start_idx < len(dataset):
            print(f"Resuming processing from sample {start_idx} out of {len(dataset)}.")
    else:
        print("No cache file found. Starting dataset preprocessing...")
        data = []
        start_idx = 0

    for idx in range(start_idx, len(dataset)):
        try:
            sample = dataset[idx]
            data.append(sample)
            if (idx + 1) % checkpoint_interval == 0:
                atomic_save(data, cache_file)
                print(
                    f"Checkpoint reached: processed {idx + 1} out of {len(dataset)} samples."
                )
        except Exception as e:
            atomic_save(data, cache_file)
            print(
                f"Error encountered while processing sample {idx}: {e}. Progress has been saved."
            )
            raise e

    torch.save(data, cache_file)
    print(f"Processing complete. Cached dataset saved to '{cache_file}'.")
    return data


def preprocess_and_cache_lazy(
    dataset,
    cache_dir: str,
    checkpoint_interval: int = 100,
):
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.pt")

    if os.path.exists(meta_path):
        print(f"Cache meta found. Loading sample paths from '{meta_path}'...")
        try:
            sample_paths = torch.load(meta_path)
        except Exception as e:
            print(f"Error loading meta: {e}. Starting from scratch.")
            sample_paths = []
        start_idx = len(sample_paths)
        if start_idx < len(dataset):
            print(f"Resuming processing at sample {start_idx} of {len(dataset)}.")
    else:
        print("No cache meta found. Starting sample-wise preprocessing...")
        sample_paths = []
        start_idx = 0

    for idx in range(start_idx, len(dataset)):
        try:
            sample = dataset[idx]
            sample_file = os.path.join(cache_dir, f"{idx:06d}.pt")
            torch.save(sample, sample_file)
            sample_paths.append(sample_file)
            if (idx + 1) % checkpoint_interval == 0:
                atomic_save(sample_paths, meta_path)
                print(f"Checkpoint reached: saved {idx+1}/{len(dataset)} samples.")
        except Exception as e:
            atomic_save(sample_paths, meta_path)
            print(f"Error at sample {idx}: {e}. Meta saved, aborting.")
            raise

    atomic_save(sample_paths, meta_path)
    print(f"Processing complete. Meta saved to '{meta_path}'.")
    return sample_paths
