import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import Optional, Tuple


def plot_spectrogram(
    spec: np.ndarray,
    sr: int = 40000,
    window_size: Optional[int] = None,
    overlap_ratio: float = 0.5,
    hop_length: Optional[int] = None,
    y_axis: str = "linear",
    x_axis: str = "time",
    cmap: str = "viridis",
    fig_size: Tuple[int, int] = (10, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Spectrogram",
    xlabel: str = "Time (s)",
    ylabel: str = "Frequency (Hz)",
    cbar_label: str = "Intensity (dB)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    if hop_length is None:
        if window_size is None:
            raise ValueError("Either 'hop_length' or 'window_size' must be specified.")
        hop_length = int(window_size * overlap_ratio)

    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis=x_axis,
        y_axis=y_axis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax
