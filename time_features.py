"""Time-domain EMG feature functions.

Each function returns a 1D array with a value per channel for the input window.
The wrapper `extract_time_features` returns concatenated [MAV, WL, ZC, SSC].
"""
from typing import Optional
import numpy as np


def mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value per channel."""
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.mean(np.abs(w), axis=0)


def wl(window: np.ndarray) -> np.ndarray:
    """Waveform length (sum of absolute differences) per channel."""
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0)


def zc(window: np.ndarray, thres: float = 0.01) -> np.ndarray:
    """Zero-crossing count per channel using an amplitude threshold.

    Counts a crossing when consecutive samples have opposite sign and
    the absolute difference exceeds `thres`.
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    prod = w[:-1, :] * w[1:, :]
    crossings = (prod < 0) & (np.abs(w[:-1, :] - w[1:, :]) >= thres)
    return np.sum(crossings, axis=0).astype(float)


def ssc(window: np.ndarray, thres: float = 0.01) -> np.ndarray:
    """Slope sign changes per channel with threshold.

    Counts where the slope changes sign around a sample and the magnitude of
    the change exceeds `thres`.
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    if w.shape[0] < 3:
        return np.zeros((w.shape[1],), dtype=float)
    # slopes around the middle sample
    s1 = w[1:-1, :] - w[:-2, :]
    s2 = w[1:-1, :] - w[2:, :]
    changes = (s1 * s2 > thres)
    return np.sum(changes, axis=0).astype(float)


def extract_time_features(window: np.ndarray, thres: Optional[float] = 0.01) -> np.ndarray:
    """Return concatenated time-domain features [MAV, WL, ZC, SSC] for a window.

    The returned array has length 4 * num_channels.
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]

    mav_v = mav(w)
    wl_v = wl(w)
    zc_v = zc(w, thres=thres)
    ssc_v = ssc(w, thres=thres)

    return np.concatenate([mav_v, wl_v, zc_v, ssc_v], axis=0)
