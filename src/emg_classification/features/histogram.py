"""Histogram-based feature extraction for EMG signals.

Implements the sEMG Histogram (HIST) feature which captures the
amplitude distribution of the EMG signal within a window.

Histogram features have been shown to perform well with non-linear
classifiers like SVM-RBF, especially for multi-class gesture recognition.

Reference: Zardoshti-Kermani et al., "EMG Feature Evaluation for Movement
Control of Upper Extremity Prostheses"
"""

from __future__ import annotations

import numpy as np


def histogram_features(
    window: np.ndarray,
    n_bins: int = 10,
    range_percentile: float = 99.0,
) -> np.ndarray:
    """Extract histogram features from an EMG window.
    
    Computes a normalized histogram of amplitude values for each channel.
    The histogram captures the distribution of signal amplitudes, which
    can be discriminative for different muscle activation patterns.
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        n_bins: Number of histogram bins per channel.
        range_percentile: Percentile for determining histogram range.
            Uses symmetric range based on this percentile of absolute values.
            
    Returns:
        1D feature vector of shape (n_bins * n_channels,).
        Each channel's histogram is normalized to sum to 1.
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n_channels = w.shape[1]
    features = []
    
    for ch in range(n_channels):
        channel_data = w[:, ch]
        
        # Determine range based on percentile of absolute values
        # This makes the histogram adaptive to the signal amplitude
        max_val = np.percentile(np.abs(channel_data), range_percentile)
        if max_val < 1e-10:
            max_val = 1.0  # Avoid zero range for silent signals
        
        # Create symmetric bins around zero
        bin_edges = np.linspace(-max_val, max_val, n_bins + 1)
        
        # Compute histogram
        hist, _ = np.histogram(channel_data, bins=bin_edges)
        
        # Normalize to get probability distribution
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        features.append(hist)
    
    return np.concatenate(features)


def histogram_features_fixed_range(
    window: np.ndarray,
    n_bins: int = 10,
    signal_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Extract histogram features with fixed amplitude range.
    
    Useful when signals have been normalized to a known range.
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        n_bins: Number of histogram bins per channel.
        signal_range: (min, max) range for histogram bins.
            
    Returns:
        1D feature vector of shape (n_bins * n_channels,).
    """
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n_channels = w.shape[1]
    features = []
    
    for ch in range(n_channels):
        channel_data = w[:, ch]
        
        # Use fixed bin edges
        bin_edges = np.linspace(signal_range[0], signal_range[1], n_bins + 1)
        
        # Compute histogram
        hist, _ = np.histogram(channel_data, bins=bin_edges)
        
        # Normalize
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        
        features.append(hist)
    
    return np.concatenate(features)


def extract_histogram_features(
    window: np.ndarray,
    n_bins: int = 10,
    adaptive_range: bool = True,
    signal_range: tuple[float, float] = (-1.0, 1.0),
    range_percentile: float = 99.0,
) -> np.ndarray:
    """Extract histogram features from an EMG window.
    
    Main entry point for histogram feature extraction.
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        n_bins: Number of histogram bins per channel.
        adaptive_range: If True, determine range from data percentiles.
            If False, use fixed signal_range.
        signal_range: (min, max) range for fixed-range mode.
        range_percentile: Percentile for adaptive range mode.
            
    Returns:
        1D feature vector of shape (n_bins * n_channels,).
    """
    if adaptive_range:
        return histogram_features(window, n_bins=n_bins, range_percentile=range_percentile)
    else:
        return histogram_features_fixed_range(window, n_bins=n_bins, signal_range=signal_range)
