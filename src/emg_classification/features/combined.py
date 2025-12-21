"""Combined feature extraction for EMG signals.

Combines spectral and time-domain features into a single feature vector.
"""

import numpy as np

from .spectral import extract_spectral_features
from .time_domain import extract_time_features


def extract_combined_features(
    window: np.ndarray,
    sampling_rate: float = 1000.0,
    threshold: float = 0.01,
    # Spectral options
    include_spectral: bool = True,
    spectral_moments: bool = True,
    spectral_flux: bool = True,
    spectral_sparsity: bool = True,
    spectral_irregularity: bool = True,
    spectral_correlation: bool = True,
    # Time domain options
    include_time: bool = True,
    time_mav: bool = True,
    time_wl: bool = True,
    time_zc: bool = True,
    time_ssc: bool = True,
    time_rms: bool = False,
    time_iemg: bool = False,
    time_var: bool = False,
) -> np.ndarray:
    """Extract combined spectral and time-domain features.
    
    Args:
        window: EMG window of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC features.
        include_spectral: Include spectral features.
        spectral_*: Options for spectral feature extraction.
        include_time: Include time-domain features.
        time_*: Options for time-domain feature extraction.
        
    Returns:
        1D feature vector combining all requested features.
    """
    if window.ndim == 1:
        window = window[:, np.newaxis]
    
    feats = []
    
    if include_spectral:
        spectral = extract_spectral_features(
            window,
            sampling_rate=sampling_rate,
            include_moments=spectral_moments,
            include_flux=spectral_flux,
            include_sparsity=spectral_sparsity,
            include_irregularity=spectral_irregularity,
            include_correlation=spectral_correlation,
        )
        feats.append(spectral)
    
    if include_time:
        time = extract_time_features(
            window,
            threshold=threshold,
            include_mav=time_mav,
            include_wl=time_wl,
            include_zc=time_zc,
            include_ssc=time_ssc,
            include_rms=time_rms,
            include_iemg=time_iemg,
            include_var=time_var,
        )
        feats.append(time)
    
    return np.concatenate(feats)


def create_feature_extractor(
    extractor_type: str = "combined",
    sampling_rate: float = 1000.0,
    **kwargs,
):
    """Create a feature extraction function with preset parameters.
    
    Args:
        extractor_type: One of 'spectral', 'time', or 'combined'.
        sampling_rate: Sampling rate in Hz.
        **kwargs: Additional arguments passed to the extractor.
        
    Returns:
        A callable that takes a window and returns features.
    """
    if extractor_type == "spectral":
        def extractor(window):
            return extract_spectral_features(window, sampling_rate=sampling_rate, **kwargs)
    elif extractor_type == "time":
        def extractor(window):
            return extract_time_features(window, **kwargs)
    elif extractor_type == "combined":
        def extractor(window):
            return extract_combined_features(window, sampling_rate=sampling_rate, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    return extractor
