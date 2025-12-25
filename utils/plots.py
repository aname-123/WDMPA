"""Plotting utils."""

import math
import os

import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter

# settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})  # set global font size
matplotlib.use("Agg")  # for writing to files only


class Colors:
    def __init__(self):
        """
        Initialize the Colors class with a pallette derived from color scheme, converting hex codes to RGB.
        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hexadecimal color `h` to an RGB tuple (PIL-compatible) with order (R, G, B)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()   # create instance for 'from utils.plots import colors'


def plot_results(file="path/to/results.csv", dir=""):
    """
    Plots training results from a 'results.csv' file; accept file path and directory as argument.

    Example: from utils.plots import *; plot_results('path/to/results.csv')
    """

    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4]):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)
                ax[i].plot(x, gaussian_filter(y, sigma=3), ":", label='smooth', linewidth=2)
                ax[i].set_title(s[j], fontsize=12)

        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[2].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Head' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')
            plt.close()

