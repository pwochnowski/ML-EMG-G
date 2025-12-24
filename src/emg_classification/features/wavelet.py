"""Wavelet-based feature extraction for EMG signals.

Implements the marginal Discrete Wavelet Transform (mDWT) features
which capture time-frequency information from EMG signals.

Wavelet features are particularly effective for capturing transient
muscle activation patterns and have shown excellent performance
in gesture recognition tasks.

Reference: Lucas et al., "Multi-channel surface EMG classification 
using support vector machines and signal-based wavelet optimization"
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np

# pywt is optional - gracefully handle if not installed
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


def check_pywt_available():
    """Check if PyWavelets is available and raise helpful error if not."""
    if not PYWT_AVAILABLE:
        raise ImportError(
            "PyWavelets (pywt) is required for wavelet features. "
            "Install it with: pip install PyWavelets"
        )


def marginal_dwt(
    window: np.ndarray,
    wavelet: str = "db7",
    level: int = 3,
) -> np.ndarray:
    """Extract marginal DWT features from an EMG window.
    
    Computes the Discrete Wavelet Transform and extracts energy
    from each decomposition level (approximation and details).
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        wavelet: Wavelet family name (e.g., 'db7', 'sym5', 'coif3').
        level: Number of decomposition levels.
        
    Returns:
        1D feature vector containing energy at each level per channel.
        Shape: ((level + 1) * n_channels,) - one for approximation + details
    """
    check_pywt_available()
    
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n_channels = w.shape[1]
    features = []
    
    for ch in range(n_channels):
        channel_data = w[:, ch]
        
        # Perform wavelet decomposition
        # Returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)
        
        # Extract energy from each level (approximation + details)
        level_energies = []
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            level_energies.append(energy)
        
        features.extend(level_energies)
    
    return np.array(features)


def marginal_dwt_normalized(
    window: np.ndarray,
    wavelet: str = "db7",
    level: int = 3,
) -> np.ndarray:
    """Extract normalized marginal DWT features.
    
    Energy is normalized by the total energy across all levels,
    giving a scale-invariant representation.
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        wavelet: Wavelet family name.
        level: Number of decomposition levels.
        
    Returns:
        1D feature vector with normalized energy per level per channel.
    """
    check_pywt_available()
    
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n_channels = w.shape[1]
    features = []
    
    for ch in range(n_channels):
        channel_data = w[:, ch]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)
        
        # Compute energy at each level
        level_energies = np.array([np.sum(coeff ** 2) for coeff in coeffs])
        
        # Normalize by total energy
        total_energy = level_energies.sum()
        if total_energy > 1e-10:
            level_energies = level_energies / total_energy
        
        features.extend(level_energies)
    
    return np.array(features)


def dwt_statistics(
    window: np.ndarray,
    wavelet: str = "db7",
    level: int = 3,
) -> np.ndarray:
    """Extract statistical features from DWT coefficients.
    
    For each decomposition level, extracts:
    - Mean absolute value of coefficients
    - Standard deviation of coefficients
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        wavelet: Wavelet family name.
        level: Number of decomposition levels.
        
    Returns:
        1D feature vector with stats per level per channel.
        Shape: (2 * (level + 1) * n_channels,)
    """
    check_pywt_available()
    
    w = window
    if w.ndim == 1:
        w = w[:, None]
    
    n_channels = w.shape[1]
    features = []
    
    for ch in range(n_channels):
        channel_data = w[:, ch]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)
        
        # Extract statistics from each level
        for coeff in coeffs:
            features.append(np.mean(np.abs(coeff)))  # MAV
            features.append(np.std(coeff))            # STD
    
    return np.array(features)


def extract_wavelet_features(
    window: np.ndarray,
    wavelet: str = "db7",
    level: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Extract wavelet features from an EMG window.
    
    Main entry point for wavelet feature extraction.
    
    Args:
        window: EMG window of shape (n_samples,) or (n_samples, n_channels).
        wavelet: Wavelet family name. Common choices:
            - 'db7': Daubechies 7 (good for EMG)
            - 'sym5': Symlet 5
            - 'coif3': Coiflet 3
        level: Number of decomposition levels. Higher = more features.
            Typical values: 3-5 depending on window size.
        normalize: If True, normalize energy by total energy (scale-invariant).
            
    Returns:
        1D feature vector of wavelet energy features.
    """
    if normalize:
        return marginal_dwt_normalized(window, wavelet=wavelet, level=level)
    else:
        return marginal_dwt(window, wavelet=wavelet, level=level)


def get_available_wavelets() -> List[str]:
    """Get list of available wavelet families.
    
    Returns:
        List of wavelet names that can be used with the extraction functions.
    """
    check_pywt_available()
    return pywt.wavelist()


def get_recommended_level(window_size: int, wavelet: str = "db7") -> int:
    """Get recommended decomposition level for a given window size.
    
    The maximum useful level depends on the signal length and wavelet filter length.
    
    Args:
        window_size: Number of samples in the window.
        wavelet: Wavelet family name.
        
    Returns:
        Recommended decomposition level.
    """
    check_pywt_available()
    
    # Get maximum level that produces non-empty coefficients
    max_level = pywt.dwt_max_level(window_size, wavelet)
    
    # Typically use 3-4 levels for EMG, but cap at max_level
    recommended = min(4, max_level)
    
    return max(1, recommended)
