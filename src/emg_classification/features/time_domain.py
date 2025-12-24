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


def lscale(window: np.ndarray) -> np.ndarray:
    """L-Scale per channel - robust measure of dispersion based on L-moments.
    
    L-Scale (λ₂) is the second L-moment and provides a robust measure of 
    statistical dispersion that is less sensitive to outliers than standard
    deviation. It is computed as half the mean absolute difference between
    pairs of order statistics.
    
    For EMG signals, L-Scale has been shown to outperform conventional 
    amplitude estimators due to its robustness to background noise spikes.
    
    Reference: Phinyomark et al., "EMG Feature Evaluation for Improving 
    Myoelectric Pattern Recognition Robustness"
    
    Args:
        window: Signal of shape (n_samples,) or (n_samples, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n = w.shape[0]
    result = np.zeros(w.shape[1])
    
    for ch in range(w.shape[1]):
        # Sort the data for each channel
        sorted_data = np.sort(w[:, ch])
        
        # Compute L-Scale using probability weighted moments
        # L2 = 2*b1 - b0, where b_r = (1/n) * sum_{i=1}^{n} ((i-1) choose r) / ((n-1) choose r) * x_{i:n}
        # Simplified formula: L2 = (1/n(n-1)) * sum_{i=1}^{n} (2i - n - 1) * x_{i:n}
        i = np.arange(1, n + 1)
        weights = (2 * i - n - 1)
        result[ch] = np.sum(weights * sorted_data) / (n * (n - 1))
    
    return result


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
    include_lscale: bool = False,
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
        include_lscale: Include L-Scale (robust dispersion).
        
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
    
    if include_lscale:
        feats.append(lscale(w))
    
    return np.concatenate(feats, axis=0)
