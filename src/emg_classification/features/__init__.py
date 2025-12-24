"""Feature extraction module for EMG signals."""

from .spectral import extract_spectral_features
from .time_domain import extract_time_features, lscale
from .combined import (
    extract_combined_features,
    extract_features_from_config,
    create_feature_extractor,
    get_feature_names,
)
from .histogram import extract_histogram_features
from .wavelet import extract_wavelet_features, PYWT_AVAILABLE

__all__ = [
    "extract_spectral_features",
    "extract_time_features",
    "extract_combined_features",
    "extract_features_from_config",
    "create_feature_extractor",
    "get_feature_names",
    "extract_histogram_features",
    "extract_wavelet_features",
    "lscale",
    "PYWT_AVAILABLE",
]
