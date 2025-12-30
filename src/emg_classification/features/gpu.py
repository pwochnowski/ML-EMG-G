"""GPU-accelerated feature extraction using CuPy.

Provides GPU-accelerated versions of spectral and time-domain feature
extraction. Falls back to CPU (NumPy) implementations if CuPy is unavailable.

Usage:
    from emg_classification.features.gpu import (
        extract_features_gpu,
        extract_features_batch_gpu,
        CUPY_AVAILABLE,
    )
    
    if CUPY_AVAILABLE:
        # Process batch on GPU
        features = extract_features_batch_gpu(windows, sampling_rate=200.0)
    else:
        # Fall back to CPU
        features = np.array([extract_combined_features(w) for w in windows])
        
To disable GPU even when CuPy is available, set:
    export EMG_DISABLE_GPU=1
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING
import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    _CUPY_IMPORTED = True
except ImportError:
    cp = None
    _CUPY_IMPORTED = False

# Allow disabling GPU via environment variable
CUPY_AVAILABLE = _CUPY_IMPORTED and not os.environ.get("EMG_DISABLE_GPU", "").lower() in ("1", "true", "yes")

if TYPE_CHECKING:
    from ..config.schema import FeatureSetConfig


def _ensure_2d(x, xp):
    """Ensure array is 2D (samples, channels)."""
    if x.ndim == 1:
        return x[:, xp.newaxis]
    return x


# =============================================================================
# GPU Time-Domain Features
# =============================================================================

def mav_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """Mean Absolute Value per channel (GPU)."""
    w = _ensure_2d(window, cp)
    return cp.mean(cp.abs(w), axis=0)


def wl_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """Waveform Length per channel (GPU)."""
    w = _ensure_2d(window, cp)
    return cp.sum(cp.abs(cp.diff(w, axis=0)), axis=0)


def zc_gpu(window: "cp.ndarray", threshold: float = 0.01) -> "cp.ndarray":
    """Zero-Crossing count per channel (GPU)."""
    w = _ensure_2d(window, cp)
    prod = w[:-1, :] * w[1:, :]
    crossings = (prod < 0) & (cp.abs(w[:-1, :] - w[1:, :]) >= threshold)
    return cp.sum(crossings, axis=0).astype(cp.float32)


def ssc_gpu(window: "cp.ndarray", threshold: float = 0.01) -> "cp.ndarray":
    """Slope Sign Changes per channel (GPU)."""
    w = _ensure_2d(window, cp)
    if w.shape[0] < 3:
        return cp.zeros((w.shape[1],), dtype=cp.float32)
    s1 = w[1:-1, :] - w[:-2, :]
    s2 = w[1:-1, :] - w[2:, :]
    changes = (s1 * s2 > threshold)
    return cp.sum(changes, axis=0).astype(cp.float32)


def rms_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """Root Mean Square per channel (GPU)."""
    w = _ensure_2d(window, cp)
    return cp.sqrt(cp.mean(w ** 2, axis=0))


def iemg_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """Integrated EMG per channel (GPU)."""
    w = _ensure_2d(window, cp)
    return cp.sum(cp.abs(w), axis=0)


def var_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """Variance per channel (GPU)."""
    w = _ensure_2d(window, cp)
    return cp.var(w, axis=0)


def lscale_gpu(window: "cp.ndarray") -> "cp.ndarray":
    """L-Scale per channel (GPU)."""
    w = _ensure_2d(window, cp)
    n = w.shape[0]
    sorted_w = cp.sort(w, axis=0)
    weights = 2 * cp.arange(1, n + 1, dtype=cp.float32) - n - 1
    return cp.sum(weights[:, cp.newaxis] * sorted_w, axis=0) / (n * (n - 1))


def extract_time_features_gpu(
    window: "cp.ndarray",
    threshold: float = 0.01,
    include_mav: bool = True,
    include_wl: bool = True,
    include_zc: bool = True,
    include_ssc: bool = True,
    include_rms: bool = False,
    include_iemg: bool = False,
    include_var: bool = False,
    include_lscale: bool = False,
) -> "cp.ndarray":
    """Extract time-domain features on GPU."""
    w = _ensure_2d(window, cp)
    feats = []
    
    if include_mav:
        feats.append(mav_gpu(w))
    if include_wl:
        feats.append(wl_gpu(w))
    if include_zc:
        feats.append(zc_gpu(w, threshold))
    if include_ssc:
        feats.append(ssc_gpu(w, threshold))
    if include_rms:
        feats.append(rms_gpu(w))
    if include_iemg:
        feats.append(iemg_gpu(w))
    if include_var:
        feats.append(var_gpu(w))
    if include_lscale:
        feats.append(lscale_gpu(w))
    
    if not feats:
        return cp.array([], dtype=cp.float32)
    return cp.concatenate(feats)


# =============================================================================
# GPU Spectral Features
# =============================================================================

def compute_fft_gpu(
    x: "cp.ndarray",
    sampling_rate: float = 1000.0,
) -> tuple["cp.ndarray", "cp.ndarray"]:
    """Compute magnitude spectrum on GPU."""
    window_size = x.shape[0]
    freqs = cp.fft.rfftfreq(window_size, 1 / sampling_rate)
    X = cp.abs(cp.fft.rfft(x, axis=0))
    return freqs, X


def spectral_moments_gpu(freqs: "cp.ndarray", X: "cp.ndarray") -> "cp.ndarray":
    """Compute spectral moments M0-M4 on GPU."""
    eps = 1e-12
    moments = []
    den = cp.sum(X, axis=0) + eps
    
    for k in range(5):
        num = cp.sum((freqs[:, cp.newaxis] ** k) * X, axis=0)
        moments.append(num / den)
    
    return cp.concatenate(moments, axis=0)


def spectral_flux_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """Compute spectral flux on GPU."""
    diff = cp.diff(X, axis=0)
    return cp.sum(diff ** 2, axis=0)


def spectral_sparsity_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """Compute spectral sparsity on GPU."""
    N = X.shape[0]
    l1 = cp.sum(cp.abs(X), axis=0)
    l2 = cp.sqrt(cp.sum(X ** 2, axis=0))
    return (cp.sqrt(N) * l1) / (l2 + 1e-12)


def irregularity_factor_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """Compute irregularity factor on GPU."""
    diff = cp.diff(X, axis=0)
    return cp.sum(diff ** 2, axis=0)


def spectrum_correlation_gpu(X: "cp.ndarray") -> "cp.ndarray":
    """Compute inter-channel power spectrum correlation on GPU."""
    P = X ** 2
    n_channels = P.shape[1]
    
    if n_channels == 1:
        return cp.array([], dtype=cp.float32)
    
    # Normalize columns
    P_norm = P - cp.mean(P, axis=0, keepdims=True)
    std = cp.std(P, axis=0, keepdims=True) + 1e-12
    P_norm = P_norm / std
    
    # Compute correlation matrix
    C = cp.dot(P_norm.T, P_norm) / P.shape[0]
    
    # Extract upper triangle
    indices = cp.triu_indices(n_channels, k=1)
    return C[indices]


def extract_spectral_features_gpu(
    window: "cp.ndarray",
    sampling_rate: float = 1000.0,
    include_moments: bool = True,
    include_flux: bool = True,
    include_sparsity: bool = True,
    include_irregularity: bool = True,
    include_correlation: bool = True,
) -> "cp.ndarray":
    """Extract spectral features on GPU."""
    w = _ensure_2d(window, cp)
    freqs, X = compute_fft_gpu(w, sampling_rate)
    
    feats = []
    
    if include_moments:
        feats.append(spectral_moments_gpu(freqs, X))
    if include_flux:
        feats.append(spectral_flux_gpu(X))
    if include_sparsity:
        feats.append(spectral_sparsity_gpu(X))
    if include_irregularity:
        feats.append(irregularity_factor_gpu(X))
    if include_correlation:
        corr = spectrum_correlation_gpu(X)
        if corr.size > 0:
            feats.append(corr)
    
    if not feats:
        return cp.array([], dtype=cp.float32)
    return cp.concatenate(feats)


# =============================================================================
# GPU Histogram Features
# =============================================================================

def extract_histogram_features_gpu(
    window: "cp.ndarray",
    n_bins: int = 10,
    adaptive_range: bool = True,
    range_percentile: float = 99.0,
) -> "cp.ndarray":
    """Extract histogram features on GPU.
    
    Matches the CPU implementation which uses percentile-based symmetric range.
    """
    w = _ensure_2d(window, cp)
    n_channels = w.shape[1]
    
    all_hists = []
    for ch in range(n_channels):
        channel_data = w[:, ch]
        if adaptive_range:
            # Match CPU: use percentile of absolute values for symmetric range
            abs_data = cp.abs(channel_data)
            max_val = float(cp.percentile(abs_data, range_percentile))
            if max_val < 1e-10:
                max_val = 1.0
            min_val = -max_val
        else:
            min_val, max_val = -1.0, 1.0
        
        hist, _ = cp.histogram(channel_data, bins=n_bins, range=(float(min_val), float(max_val)))
        hist = hist.astype(cp.float32)
        # Normalize by sum (not total samples) to get proper probability distribution
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        all_hists.append(hist)
    
    return cp.concatenate(all_hists)


# =============================================================================
# Combined GPU Feature Extraction
# =============================================================================

def extract_combined_features_gpu(
    window: "cp.ndarray",
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
) -> "cp.ndarray":
    """Extract combined features on GPU.
    
    Args:
        window: CuPy array of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC features.
        ... (same options as CPU version)
        
    Returns:
        CuPy 1D array of features.
    """
    w = _ensure_2d(window, cp)
    feats = []
    
    if include_spectral:
        spectral = extract_spectral_features_gpu(
            w,
            sampling_rate=sampling_rate,
            include_moments=spectral_moments,
            include_flux=spectral_flux,
            include_sparsity=spectral_sparsity,
            include_irregularity=spectral_irregularity,
            include_correlation=spectral_correlation,
        )
        feats.append(spectral)
    
    if include_time:
        time = extract_time_features_gpu(
            w,
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
    
    if include_histogram:
        hist = extract_histogram_features_gpu(w, n_bins=hist_bins)
        feats.append(hist)
    
    if not feats:
        raise ValueError("No features selected!")
    
    return cp.concatenate(feats)


def extract_features_from_config_gpu(
    window: "cp.ndarray",
    config: "FeatureSetConfig",
    sampling_rate: float = 1000.0,
) -> "cp.ndarray":
    """Extract features using a FeatureSetConfig on GPU.
    
    Note: Wavelet features are not yet GPU-accelerated and will be skipped.
    """
    include_spectral = any([
        config.spectral_moments,
        config.spectral_flux,
        config.spectral_sparsity,
        config.spectral_irregularity,
        config.spectral_correlation,
    ])
    
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
    
    if config.wavelet_enabled:
        import warnings
        warnings.warn("Wavelet features not yet GPU-accelerated, will be skipped")
    
    return extract_combined_features_gpu(
        window,
        sampling_rate=sampling_rate,
        threshold=config.threshold,
        include_spectral=include_spectral,
        spectral_moments=config.spectral_moments,
        spectral_flux=config.spectral_flux,
        spectral_sparsity=config.spectral_sparsity,
        spectral_irregularity=config.spectral_irregularity,
        spectral_correlation=config.spectral_correlation,
        include_time=include_time,
        time_mav=config.time_mav,
        time_wl=config.time_wl,
        time_zc=config.time_zc,
        time_ssc=config.time_ssc,
        time_rms=config.time_rms,
        time_iemg=config.time_iemg,
        time_var=config.time_var,
        time_lscale=config.time_lscale,
        include_histogram=config.hist_enabled,
        hist_bins=config.hist_bins,
    )


# =============================================================================
# Batch Processing (Main GPU Advantage)
# =============================================================================

def extract_features_batch_gpu(
    windows: np.ndarray,
    sampling_rate: float = 1000.0,
    threshold: float = 0.01,
    config: Optional["FeatureSetConfig"] = None,
    return_numpy: bool = True,
) -> np.ndarray:
    """Extract features from multiple windows in batch on GPU.
    
    This is the main function to use for GPU-accelerated feature extraction.
    Processing in batch is much more efficient than one window at a time.
    
    Args:
        windows: NumPy array of shape (n_windows, window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC.
        config: Optional FeatureSetConfig for custom features.
        return_numpy: If True, return NumPy array; if False, return CuPy array.
        
    Returns:
        Feature matrix of shape (n_windows, n_features).
        
    Example:
        >>> windows = np.random.randn(1000, 200, 8)  # 1000 windows, 200 samples, 8 channels
        >>> features = extract_features_batch_gpu(windows, sampling_rate=200.0)
        >>> print(features.shape)  # (1000, n_features)
    """
    if not CUPY_AVAILABLE:
        raise ImportError(
            "CuPy is required for GPU feature extraction. "
            "Install with: uv sync --extra gpu"
        )
    
    import logging
    logger = logging.getLogger(__name__)
    
    n_windows = windows.shape[0]
    logger.info(f"  GPU batch: {n_windows} windows, shape {windows.shape}")
    
    # Process in smaller chunks and sync after each to detect GPU errors early
    chunk_size = 500  # Smaller chunks
    all_features = []
    
    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_windows)
        chunk_windows = windows[chunk_start:chunk_end]
        n_chunk = chunk_end - chunk_start
        
        logger.info(f"  GPU batch: Processing chunk {chunk_start}-{chunk_end} of {n_windows} ({100*chunk_start//n_windows}%)")
        
        try:
            # Transfer chunk to GPU
            windows_gpu = cp.asarray(chunk_windows, dtype=cp.float32)
            
            # Extract features for each window in chunk
            features_list = []
            for i in range(n_chunk):
                if config is not None:
                    feat = extract_features_from_config_gpu(windows_gpu[i], config, sampling_rate)
                else:
                    feat = extract_combined_features_gpu(
                        windows_gpu[i],
                        sampling_rate=sampling_rate,
                        threshold=threshold,
                    )
                features_list.append(feat)
            
            # Stack chunk features and transfer to CPU immediately
            chunk_features_gpu = cp.stack(features_list, axis=0)
            
            # Synchronize to catch any GPU errors
            cp.cuda.Stream.null.synchronize()
            
            chunk_features = cp.asnumpy(chunk_features_gpu)
            all_features.append(chunk_features)
            
            # Explicit memory cleanup
            del windows_gpu, features_list, chunk_features_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
        except cp.cuda.runtime.CUDARuntimeError as e:
            logger.error(f"  GPU error at chunk {chunk_start}: {e}")
            logger.info("  Falling back to CPU for remaining windows...")
            # Fall back to CPU for this and remaining chunks
            from .combined import extract_combined_features, extract_features_from_config
            for j in range(chunk_start, n_windows):
                w = windows[j]
                if config is not None:
                    feat = extract_features_from_config(w, config, sampling_rate)
                else:
                    feat = extract_combined_features(w, sampling_rate=sampling_rate, threshold=threshold)
                all_features.append(feat.reshape(1, -1))
            break
    
    # Combine all chunks
    result = np.vstack(all_features)
    logger.info(f"  GPU batch: Complete, result shape {result.shape}")
    
    if return_numpy:
        return result
    return cp.asarray(result)


def extract_features_gpu(
    window: np.ndarray,
    sampling_rate: float = 1000.0,
    threshold: float = 0.01,
    config: Optional["FeatureSetConfig"] = None,
    return_numpy: bool = True,
) -> np.ndarray:
    """Extract features from a single window on GPU.
    
    For single windows, GPU overhead may make this slower than CPU.
    Use extract_features_batch_gpu for batch processing.
    
    Args:
        window: NumPy array of shape (window_size, n_channels).
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC.
        config: Optional FeatureSetConfig.
        return_numpy: If True, return NumPy array.
        
    Returns:
        1D feature vector.
    """
    if not CUPY_AVAILABLE:
        raise ImportError(
            "CuPy is required for GPU feature extraction. "
            "Install with: uv sync --extra gpu"
        )
    
    window_gpu = cp.asarray(window, dtype=cp.float32)
    
    if config is not None:
        features = extract_features_from_config_gpu(window_gpu, config, sampling_rate)
    else:
        features = extract_combined_features_gpu(
            window_gpu,
            sampling_rate=sampling_rate,
            threshold=threshold,
        )
    
    if return_numpy:
        return cp.asnumpy(features)
    return features


# =============================================================================
# Auto-dispatch function (GPU if available, else CPU)
# =============================================================================

def extract_features_auto(
    windows: np.ndarray,
    sampling_rate: float = 1000.0,
    threshold: float = 0.01,
    config: Optional["FeatureSetConfig"] = None,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """Automatically use GPU if available, otherwise CPU.
    
    Args:
        windows: Array of shape (n_windows, window_size, n_channels) or 
                 (window_size, n_channels) for single window.
        sampling_rate: Sampling rate in Hz.
        threshold: Threshold for ZC and SSC.
        config: Optional FeatureSetConfig.
        prefer_gpu: If True and GPU available, use GPU.
        
    Returns:
        Feature matrix/vector.
    """
    single_window = windows.ndim == 2
    if single_window:
        windows = windows[np.newaxis, ...]
    
    if prefer_gpu and CUPY_AVAILABLE:
        features = extract_features_batch_gpu(
            windows,
            sampling_rate=sampling_rate,
            threshold=threshold,
            config=config,
        )
    else:
        # CPU fallback
        from .combined import extract_combined_features, extract_features_from_config
        
        features_list = []
        for i in range(windows.shape[0]):
            if config is not None:
                feat = extract_features_from_config(windows[i], config, sampling_rate)
            else:
                feat = extract_combined_features(
                    windows[i],
                    sampling_rate=sampling_rate,
                    threshold=threshold,
                )
            features_list.append(feat)
        features = np.array(features_list)
    
    if single_window:
        return features[0]
    return features
