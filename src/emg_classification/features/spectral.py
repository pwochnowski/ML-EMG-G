"""Spectral feature extraction for EMG signals.

Based on Khushaba et al. spectral features including spectral moments,
flux, sparsity, irregularity factor, and power spectrum correlation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def compute_fft(
    x: np.ndarray,
    sampling_rate: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum for each channel.
    
    Args:
        x: Signal window of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        
    Returns:
        Tuple of (frequencies, magnitude_spectrum).
        - frequencies: Array of frequency bins.
        - magnitude_spectrum: Magnitude at each frequency, shape (n_freqs, n_channels).
    """
    window_size = x.shape[0]
    freqs = np.fft.rfftfreq(window_size, 1 / sampling_rate)
    X = np.abs(np.fft.rfft(x, axis=0))
    return freqs, X


def spectral_moments(freqs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute spectral moments M0-M4 for each channel.
    
    Args:
        freqs: Frequency bins.
        X: Magnitude spectrum of shape (n_freqs, n_channels).
        
    Returns:
        Array of shape (5 * n_channels,) containing M0-M4 for each channel.
    """
    eps = 1e-12
    moments = []
    
    for k in range(5):
        num = np.sum((freqs[:, None] ** k) * X, axis=0)
        den = np.sum(X, axis=0) + eps
        moments.append(num / den)
    
    return np.concatenate(moments, axis=0)


def spectral_flux(X: np.ndarray) -> np.ndarray:
    """Compute spectral flux for each channel.
    
    Measures the rate of change of the spectrum.
    
    Args:
        X: Magnitude spectrum of shape (n_freqs, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    diff = np.diff(X, axis=0)
    flux = np.sum(diff ** 2, axis=0)
    return flux


def spectral_sparsity(X: np.ndarray) -> np.ndarray:
    """Compute spectral sparsity for each channel.
    
    Measures how concentrated the spectrum is.
    
    Args:
        X: Magnitude spectrum of shape (n_freqs, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    N = X.shape[0]
    l1 = np.sum(np.abs(X), axis=0)
    l2 = np.sqrt(np.sum(X ** 2, axis=0))
    return (np.sqrt(N) * l1) / (l2 + 1e-12)


def irregularity_factor(X: np.ndarray) -> np.ndarray:
    """Compute irregularity factor for each channel.
    
    Measures the jaggedness of the spectrum.
    
    Args:
        X: Magnitude spectrum of shape (n_freqs, n_channels).
        
    Returns:
        Array of shape (n_channels,).
    """
    diff = np.diff(X, axis=0)
    return np.sum(diff ** 2, axis=0)


def spectrum_correlation(
    X: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute power spectrum correlation.
    
    If reference is None, computes inter-channel correlation (upper triangle).
    Otherwise, computes correlation of each channel with the reference.
    
    Args:
        X: Magnitude spectrum of shape (n_freqs, n_channels).
        reference: Optional reference spectrum of shape (n_freqs,).
        
    Returns:
        Correlation values. Shape depends on mode:
        - With reference: (n_channels,)
        - Without reference: (n_channels * (n_channels - 1) / 2,)
    """
    P = X ** 2  # power spectrum
    
    if reference is not None:
        # Correlation with reference spectrum
        corr = []
        for ch in range(X.shape[1]):
            corr.append(np.corrcoef(P[:, ch], reference)[0, 1])
        return np.array(corr)
    
    # Inter-channel correlation (upper triangle)
    C = np.corrcoef(P.T)
    return C[np.triu_indices_from(C, k=1)]


def extract_spectral_features(
    window: np.ndarray,
    sampling_rate: float = 1000.0,
    reference_spectrum: Optional[np.ndarray] = None,
    include_moments: bool = True,
    include_flux: bool = True,
    include_sparsity: bool = True,
    include_irregularity: bool = True,
    include_correlation: bool = True,
) -> np.ndarray:
    """Extract full spectral feature vector from an EMG window.
    
    Args:
        window: EMG window of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        reference_spectrum: Optional reference spectrum for correlation.
        include_moments: Include spectral moments M0-M4.
        include_flux: Include spectral flux.
        include_sparsity: Include spectral sparsity.
        include_irregularity: Include irregularity factor.
        include_correlation: Include spectrum correlation.
        
    Returns:
        1D feature vector. Length depends on options and n_channels.
        Default: 5*ch + ch + ch + ch + ch*(ch-1)/2 features.
    """
    if window.ndim == 1:
        window = window[:, np.newaxis]
    
    freqs, X = compute_fft(window, sampling_rate)
    
    feats = []
    
    if include_moments:
        feats.append(spectral_moments(freqs, X))
    
    if include_flux:
        feats.append(spectral_flux(X))
    
    if include_sparsity:
        feats.append(spectral_sparsity(X))
    
    if include_irregularity:
        feats.append(irregularity_factor(X))
    
    if include_correlation:
        feats.append(spectrum_correlation(X, reference_spectrum))
    
    return np.concatenate(feats)


# Alias for backwards compatibility
extract_khushaba_features = extract_spectral_features
