"""Time-domain feature extraction for EMG signals.

Implements standard time-domain EMG features: MAV, WL, ZC, SSC.
"""

from typing import Optional
import numpy as np


def mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value per channel.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.mean(np.abs(w), axis=0)


def wl(window: np.ndarray) -> np.ndarray:
    """Waveform Length (sum of absolute differences) per channel.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0)


def zc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Zero-Crossing count per channel with amplitude threshold.
    
    Counts crossings where consecutive samples have opposite signs
    and the absolute difference exceeds the threshold.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        threshold: Minimum amplitude change to count as crossing.
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    prod = w[:-1, :] * w[1:, :]
    crossings = (prod < 0) & (np.abs(w[:-1, :] - w[1:, :]) >= threshold)
    return np.sum(crossings, axis=0).astype(float)


def ssc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Slope Sign Changes per channel with threshold.
    
    Counts where the slope changes sign and the magnitude exceeds threshold.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        threshold: Minimum slope product to count as change.
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    if w.shape[0] < 3:
        return np.zeros((w.shape[1],), dtype=float)
    
    # Slopes around the middle sample
    s1 = w[1:-1, :] - w[:-2, :]
    s2 = w[1:-1, :] - w[2:, :]
    changes = (s1 * s2 > threshold)
    return np.sum(changes, axis=0).astype(float)


def rms(window: np.ndarray) -> np.ndarray:
    """Root Mean Square per channel.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.sqrt(np.mean(w ** 2, axis=0))


def iemg(window: np.ndarray) -> np.ndarray:
    """Integrated EMG (sum of absolute values) per channel.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.sum(np.abs(w), axis=0)


def var(window: np.ndarray) -> np.ndarray:
    """Variance per channel.
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    return np.var(w, axis=0)


def extract_time_features(
    window: np.ndarray,
    threshold: Optional[float] = 0.01,
    include_mav: bool = True,
    include_wl: bool = True,
    include_zc: bool = True,
    include_ssc: bool = True,
    include_rms: bool = False,
    include_iemg: bool = False,
    include_var: bool = False,
) -> np.ndarray:
    """Extract time-domain features from an EMG window.
    
    Default features are [MAV, WL, ZC, SSC] for each channel.
    
    Args:
        window: EMG window of shape (window_size, n_channels).
        threshold: Threshold for ZC and SSC features.
        include_mav: Include Mean Absolute Value.
        include_wl: Include Waveform Length.
        include_zc: Include Zero Crossings.
        include_ssc: Include Slope Sign Changes.
        include_rms: Include Root Mean Square.
        include_iemg: Include Integrated EMG.
        include_var: Include Variance.
        
    Returns:
        1D feature vector. Length = (number of features) * n_channels.
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    feats = []
    
    if include_mav:
        feats.append(mav(w))
    
    if include_wl:
        feats.append(wl(w))
    
    if include_zc:
        feats.append(zc(w, threshold=threshold))
    
    if include_ssc:
        feats.append(ssc(w, threshold=threshold))
    
    if include_rms:
        feats.append(rms(w))
    
    if include_iemg:
        feats.append(iemg(w))
    
    if include_var:
        feats.append(var(w))
    
    return np.concatenate(feats, axis=0)
