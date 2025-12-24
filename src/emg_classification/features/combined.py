"""Combined feature extraction for EMG signals.

Combines spectral, time-domain, histogram, and wavelet features 
into a single feature vector. Supports configuration via FeatureSetConfig.
"""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING
import numpy as np

from .spectral import extract_spectral_features
from .time_domain import extract_time_features
from .histogram import extract_histogram_features
from .wavelet import extract_wavelet_features, PYWT_AVAILABLE

if TYPE_CHECKING:
    from ..config.schema import FeatureSetConfig


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
    time_lscale: bool = False,
    # Histogram options
    include_histogram: bool = False,
    hist_bins: int = 10,
    # Wavelet options
    include_wavelet: bool = False,
    wavelet_name: str = "db7",
    wavelet_level: int = 3,
) -> np.ndarray:
    """Extract combined features from an EMG window.
    
    Args:
        window: EMG window of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC features.
        include_spectral: Include spectral features.
        spectral_*: Options for spectral feature extraction.
        include_time: Include time-domain features.
        time_*: Options for time-domain feature extraction.
        include_histogram: Include histogram features.
        hist_bins: Number of histogram bins.
        include_wavelet: Include wavelet features.
        wavelet_name: Wavelet family name.
        wavelet_level: Wavelet decomposition levels.
        
    Returns:
        1D feature vector combining all requested features.
    """
    if window.ndim == 1:
        window = window[:, np.newaxis]
    
    feats = []
    
    # Spectral features
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
    
    # Time-domain features
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
            include_lscale=time_lscale,
        )
        feats.append(time)
    
    # Histogram features
    if include_histogram:
        hist = extract_histogram_features(
            window,
            n_bins=hist_bins,
            adaptive_range=True,
        )
        feats.append(hist)
    
    # Wavelet features
    if include_wavelet:
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets (pywt) is required for wavelet features. "
                "Install it with: pip install PyWavelets"
            )
        wavelet = extract_wavelet_features(
            window,
            wavelet=wavelet_name,
            level=wavelet_level,
            normalize=True,
        )
        feats.append(wavelet)
    
    if not feats:
        raise ValueError("No features selected! Enable at least one feature type.")
    
    return np.concatenate(feats)


def extract_features_from_config(
    window: np.ndarray,
    config: "FeatureSetConfig",
    sampling_rate: float = 1000.0,
) -> np.ndarray:
    """Extract features using a FeatureSetConfig.
    
    This is the recommended way to extract features when using
    named feature sets from features.yaml.
    
    Args:
        window: EMG window of shape (window_size, n_channels).
        config: FeatureSetConfig loaded from features.yaml.
        sampling_rate: Sampling rate in Hz.
        
    Returns:
        1D feature vector.
        
    Example:
        >>> from emg_classification.config.schema import load_feature_set
        >>> config = load_feature_set("experimental")
        >>> features = extract_features_from_config(window, config, sampling_rate=200.0)
    """
    # Determine if any spectral features are enabled
    include_spectral = any([
        config.spectral_moments,
        config.spectral_flux,
        config.spectral_sparsity,
        config.spectral_irregularity,
        config.spectral_correlation,
    ])
    
    # Determine if any time-domain features are enabled
    include_time = any([
        config.time_mav,
        config.time_wl,
        config.time_zc,
        config.time_ssc,
        config.time_rms,
        config.time_iemg,
        config.time_var,
        config.time_lscale,
    ])
    
    return extract_combined_features(
        window,
        sampling_rate=sampling_rate,
        threshold=config.threshold,
        # Spectral
        include_spectral=include_spectral,
        spectral_moments=config.spectral_moments,
        spectral_flux=config.spectral_flux,
        spectral_sparsity=config.spectral_sparsity,
        spectral_irregularity=config.spectral_irregularity,
        spectral_correlation=config.spectral_correlation,
        # Time-domain
        include_time=include_time,
        time_mav=config.time_mav,
        time_wl=config.time_wl,
        time_zc=config.time_zc,
        time_ssc=config.time_ssc,
        time_rms=config.time_rms,
        time_iemg=config.time_iemg,
        time_var=config.time_var,
        time_lscale=config.time_lscale,
        # Histogram
        include_histogram=config.hist_enabled,
        hist_bins=config.hist_bins,
        # Wavelet
        include_wavelet=config.wavelet_enabled,
        wavelet_name=config.wavelet_name,
        wavelet_level=config.wavelet_level,
    )


def create_feature_extractor(
    extractor_type: str = "combined",
    sampling_rate: float = 1000.0,
    config: Optional["FeatureSetConfig"] = None,
    **kwargs,
):
    """Create a feature extraction function with preset parameters.
    
    Args:
        extractor_type: One of 'spectral', 'time', 'combined', or 'config'.
        sampling_rate: Sampling rate in Hz.
        config: Optional FeatureSetConfig for 'config' mode or to override 'combined'.
        **kwargs: Additional arguments passed to the extractor.
        
    Returns:
        A callable that takes a window and returns features.
        
    Example:
        >>> # Using a named feature set
        >>> from emg_classification.config.schema import load_feature_set
        >>> config = load_feature_set("experimental")
        >>> extractor = create_feature_extractor(config=config, sampling_rate=200.0)
        >>> features = extractor(window)
    """
    # If config is provided, use config-based extraction
    if config is not None:
        def extractor(window):
            return extract_features_from_config(window, config, sampling_rate=sampling_rate)
        return extractor
    
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


def get_feature_names(
    config: "FeatureSetConfig",
    n_channels: int,
) -> list[str]:
    """Get the names of all features that will be extracted.
    
    Useful for understanding feature importance and debugging.
    
    Args:
        config: FeatureSetConfig specifying which features are enabled.
        n_channels: Number of EMG channels.
        
    Returns:
        List of feature names in the order they appear in the feature vector.
    """
    names = []
    
    # Spectral features
    if config.spectral_moments:
        for m in range(5):
            for ch in range(n_channels):
                names.append(f"spectral_m{m}_ch{ch}")
    if config.spectral_flux:
        for ch in range(n_channels):
            names.append(f"spectral_flux_ch{ch}")
    if config.spectral_sparsity:
        for ch in range(n_channels):
            names.append(f"spectral_sparsity_ch{ch}")
    if config.spectral_irregularity:
        for ch in range(n_channels):
            names.append(f"spectral_irregularity_ch{ch}")
    if config.spectral_correlation:
        # Inter-channel correlation (upper triangle)
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                names.append(f"spectral_corr_ch{i}_ch{j}")
    
    # Time-domain features
    time_features = [
        ("mav", config.time_mav),
        ("wl", config.time_wl),
        ("zc", config.time_zc),
        ("ssc", config.time_ssc),
        ("rms", config.time_rms),
        ("iemg", config.time_iemg),
        ("var", config.time_var),
        ("lscale", config.time_lscale),
    ]
    for feat_name, enabled in time_features:
        if enabled:
            for ch in range(n_channels):
                names.append(f"time_{feat_name}_ch{ch}")
    
    # Histogram features
    if config.hist_enabled:
        for ch in range(n_channels):
            for b in range(config.hist_bins):
                names.append(f"hist_bin{b}_ch{ch}")
    
    # Wavelet features
    if config.wavelet_enabled:
        for ch in range(n_channels):
            names.append(f"wavelet_approx_ch{ch}")
            for lvl in range(config.wavelet_level):
                names.append(f"wavelet_detail{lvl+1}_ch{ch}")
    
    return names
