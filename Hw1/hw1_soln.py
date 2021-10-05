import argparse
import os
import pathlib
from typing import Tuple


import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import sympy

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def get_2nd_largest(x, axis=1):
    """
    Get the index of second largest energy value in data
    """
    ind = np.argmax(abs(x), axis) + 1
    return ind


def plot_hist(
    input, n_bins: int, ranges: Tuple[int, int], title: str, save_path: str = ""
):
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [plt.close() if event.key in ["escape", "Q"] else None],
    )
    count = plt.hist(input, bins=n_bins, range=ranges)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = current_dir / "data"
    mat_fname = data_dir / "TrainingSamplesDCT_8.mat"
    zig_fname = data_dir / "Zig-Zag Pattern.txt"
    plot_dir = current_dir / "plots"

    for d in [data_dir, plot_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    
    zig_zag = np.loadtxt(zig_fname)
    ic(zig_zag.shape)

    mat_contents = sio.loadmat(mat_fname)
    TrainsampleDCT_BG = mat_contents["TrainsampleDCT_BG"]
    TrainsampleDCT_FG = mat_contents["TrainsampleDCT_FG"]
    ic(TrainsampleDCT_BG.shape)
    ic(TrainsampleDCT_FG.shape)

    m_cheetah, n_cheetah = np.shape(TrainsampleDCT_BG)
    m_grass, n_grass = np.shape(TrainsampleDCT_FG)

    # (a)
    P_cheetah = m_cheetah / (m_cheetah + m_grass)
    P_grass = m_grass / (m_cheetah + m_grass)

    assert P_cheetah + P_grass == 1
    print(f"The prior P_Y_cheetah: {P_cheetah}")
    print(f"The prior P_Y_grass: {P_grass}")

    # (b)
    cheetah_index = get_2nd_largest(TrainsampleDCT_FG[:, 1:])
    grass_index = get_2nd_largest(TrainsampleDCT_BG[:, 1:])

    plot_hist(
        cheetah_index,
        n_bins=n_cheetah,
        ranges=(0, n_cheetah - 1),
        title="Histogram of Cheetah",
        save_path=plot_dir / "hist_FG",
    )
    plot_hist(
        grass_index,
        n_bins=n_grass,
        ranges=(0, n_grass - 1),
        title="Histogram of Grass",
        save_path=plot_dir / "hist_BG",
    )
