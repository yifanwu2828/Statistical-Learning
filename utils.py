from typing import Tuple
import pathlib

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt


sqrt_2_PI = np.sqrt(2 * np.pi)
current_dir = pathlib.Path(__file__).parent.resolve()
zigzag = np.loadtxt(current_dir /"Zig-Zag Pattern.txt", dtype=np.int64)


def dct2(block: np.ndarray) -> np.ndarray:
    """
    Compute the DCT2 of the data.
    """
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def im2double(img: np.ndarray) -> np.ndarray:
    """
    Converts the image to double.
    """
    return img.astype(np.float64) / 255

def padding(img: np.ndarray, pad_size: int) -> np.ndarray:
    """
    Pads the image with zeros.
    """
    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), "constant")


def imagesc(img: np.ndarray, title: str = "imagesc Segmented Image") -> None:
    # equavalent to imagesc
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=[-1, 1, -1, 1])
    plt.title(title)
    plt.show()

def colormap_gray255(img: np.ndarray, title: str = "Grayscale Segmented Image") -> None:
    """equvalent to colormap(gray(255))"""
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.show()

def calcSigma(data: np.ndarray, bias = False) -> np.ndarray:
    """
    Calculate the covariance of the data.
    assume the data is a 2D array and column represents a variable.
    
    """
    m_samples = data.shape[0]
    n_features = data.shape[1]
    mean = np.mean(data, axis=0)
    
    cov = np.zeros([n_features, n_features], dtype=np.float64)
    for k in range(m_samples):
        distance = (data[k, :] - mean).reshape(64, 1)
        cov += distance @ distance.T
    if bias:
        return cov / m_samples
    else:
        return cov / (m_samples - 1)
    
def calculate_error(A: np.ndarray, ground_truth: np.ndarray, verbose=True) -> Tuple[float, float, float]:
    """
    compute the probability of error by comparing with cheetah mask.bmp.
    """
    # Truncate ground truth to have same size as segmented image
    ground_truth = ground_truth[: A.shape[0], : A.shape[1]] / 255
    
    # calculate the error
    error = 1 - np.sum(ground_truth == A) / A.size

    # error in the FG
    error_idex = np.where((ground_truth - A) == 1)[0]
    FG_error = len(error_idex) / A.size
   
    # error in the BG
    error_idex = np.where((ground_truth - A) == -1)[0]
    BG_error = len(error_idex) / A.size
    if verbose:
        print(f"The probability of error: {error}")
        print(f"FG error: {FG_error}")
        print(f"BG error is: {BG_error}")
    
    return error, FG_error, BG_error



