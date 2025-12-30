"""Tests for GPU-accelerated feature extraction.

Compares GPU (CuPy) implementations against CPU (NumPy) implementations
to ensure numerical consistency.
"""

import numpy as np
import pytest

# Import CPU implementations
from src.emg_classification.features.time_domain import (
    mav, wl, zc, ssc, rms, iemg, var, lscale,
    extract_time_features,
)
from src.emg_classification.features.spectral import (
    compute_fft, spectral_moments, spectral_flux, spectral_sparsity,
    irregularity_factor, spectrum_correlation, extract_spectral_features,
)
from src.emg_classification.features.histogram import extract_histogram_features
from src.emg_classification.features.combined import (
    extract_combined_features, extract_features_from_config,
)

# Try to import GPU implementations
try:
    import cupy as cp
    from src.emg_classification.features.gpu import (
        CUPY_AVAILABLE,
        mav_gpu, wl_gpu, zc_gpu, ssc_gpu, rms_gpu, iemg_gpu, var_gpu, lscale_gpu,
        extract_time_features_gpu,
        compute_fft_gpu, spectral_moments_gpu, spectral_flux_gpu,
        spectral_sparsity_gpu, irregularity_factor_gpu, spectrum_correlation_gpu,
        extract_spectral_features_gpu,
        extract_histogram_features_gpu,
        extract_combined_features_gpu,
        extract_features_from_config_gpu,
        extract_features_batch_gpu,
        extract_features_gpu,
    )
    GPU_AVAILABLE = CUPY_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")


# Test fixtures
@pytest.fixture
def single_channel_window():
    """Single channel EMG window."""
    np.random.seed(42)
    return np.random.randn(200).astype(np.float32)


@pytest.fixture
def multi_channel_window():
    """Multi-channel EMG window (8 channels)."""
    np.random.seed(42)
    return np.random.randn(200, 8).astype(np.float32)


@pytest.fixture
def batch_windows():
    """Batch of windows for batch processing tests."""
    np.random.seed(42)
    return np.random.randn(50, 200, 8).astype(np.float32)


@pytest.fixture
def feature_config():
    """Load default feature config."""
    from src.emg_classification.config.schema import load_feature_set
    return load_feature_set("default")


# =============================================================================
# Time-domain feature tests
# =============================================================================

class TestTimeDomainFeatures:
    """Test time-domain feature implementations."""
    
    def test_mav(self, multi_channel_window):
        """Test Mean Absolute Value."""
        cpu_result = mav(multi_channel_window)
        gpu_result = cp.asnumpy(mav_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_wl(self, multi_channel_window):
        """Test Waveform Length."""
        cpu_result = wl(multi_channel_window)
        gpu_result = cp.asnumpy(wl_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_zc(self, multi_channel_window):
        """Test Zero Crossings."""
        cpu_result = zc(multi_channel_window, threshold=0.01)
        gpu_result = cp.asnumpy(zc_gpu(cp.asarray(multi_channel_window), threshold=0.01))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_ssc(self, multi_channel_window):
        """Test Slope Sign Changes."""
        cpu_result = ssc(multi_channel_window, threshold=0.01)
        gpu_result = cp.asnumpy(ssc_gpu(cp.asarray(multi_channel_window), threshold=0.01))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_rms(self, multi_channel_window):
        """Test Root Mean Square."""
        cpu_result = rms(multi_channel_window)
        gpu_result = cp.asnumpy(rms_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_iemg(self, multi_channel_window):
        """Test Integrated EMG."""
        cpu_result = iemg(multi_channel_window)
        gpu_result = cp.asnumpy(iemg_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_var(self, multi_channel_window):
        """Test Variance."""
        cpu_result = var(multi_channel_window)
        gpu_result = cp.asnumpy(var_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_lscale(self, multi_channel_window):
        """Test L-Scale."""
        cpu_result = lscale(multi_channel_window)
        gpu_result = cp.asnumpy(lscale_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-5)
    
    def test_extract_time_features_default(self, multi_channel_window):
        """Test combined time feature extraction with default options."""
        cpu_result = extract_time_features(multi_channel_window)
        gpu_result = cp.asnumpy(extract_time_features_gpu(cp.asarray(multi_channel_window)))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_extract_time_features_all(self, multi_channel_window):
        """Test combined time feature extraction with all features enabled."""
        cpu_result = extract_time_features(
            multi_channel_window,
            include_mav=True, include_wl=True, include_zc=True, include_ssc=True,
            include_rms=True, include_iemg=True, include_var=True, include_lscale=True,
        )
        gpu_result = cp.asnumpy(extract_time_features_gpu(
            cp.asarray(multi_channel_window),
            include_mav=True, include_wl=True, include_zc=True, include_ssc=True,
            include_rms=True, include_iemg=True, include_var=True, include_lscale=True,
        ))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-5)


# =============================================================================
# Spectral feature tests
# =============================================================================

class TestSpectralFeatures:
    """Test spectral feature implementations."""
    
    def test_compute_fft(self, multi_channel_window):
        """Test FFT computation."""
        freqs_cpu, X_cpu = compute_fft(multi_channel_window, sampling_rate=200.0)
        freqs_gpu, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        np.testing.assert_allclose(freqs_cpu, cp.asnumpy(freqs_gpu), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(X_cpu, cp.asnumpy(X_gpu), rtol=1e-5, atol=1e-6)
    
    def test_spectral_moments(self, multi_channel_window):
        """Test spectral moments M0-M4."""
        freqs, X = compute_fft(multi_channel_window, sampling_rate=200.0)
        freqs_gpu, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        cpu_result = spectral_moments(freqs, X)
        gpu_result = cp.asnumpy(spectral_moments_gpu(freqs_gpu, X_gpu))
        
        # Higher moments have larger values, so use relative tolerance
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)
    
    def test_spectral_flux(self, multi_channel_window):
        """Test spectral flux."""
        _, X = compute_fft(multi_channel_window, sampling_rate=200.0)
        _, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        cpu_result = spectral_flux(X)
        gpu_result = cp.asnumpy(spectral_flux_gpu(X_gpu))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_spectral_sparsity(self, multi_channel_window):
        """Test spectral sparsity."""
        _, X = compute_fft(multi_channel_window, sampling_rate=200.0)
        _, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        cpu_result = spectral_sparsity(X)
        gpu_result = cp.asnumpy(spectral_sparsity_gpu(X_gpu))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_irregularity_factor(self, multi_channel_window):
        """Test irregularity factor."""
        _, X = compute_fft(multi_channel_window, sampling_rate=200.0)
        _, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        cpu_result = irregularity_factor(X)
        gpu_result = cp.asnumpy(irregularity_factor_gpu(X_gpu))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_spectrum_correlation(self, multi_channel_window):
        """Test inter-channel spectrum correlation."""
        _, X = compute_fft(multi_channel_window, sampling_rate=200.0)
        _, X_gpu = compute_fft_gpu(cp.asarray(multi_channel_window), sampling_rate=200.0)
        
        cpu_result = spectrum_correlation(X)
        gpu_result = cp.asnumpy(spectrum_correlation_gpu(X_gpu))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)
    
    def test_extract_spectral_features(self, multi_channel_window):
        """Test combined spectral feature extraction."""
        cpu_result = extract_spectral_features(multi_channel_window, sampling_rate=200.0)
        gpu_result = cp.asnumpy(extract_spectral_features_gpu(
            cp.asarray(multi_channel_window), sampling_rate=200.0
        ))
        
        # Use slightly looser tolerance due to accumulated floating point differences
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)


# =============================================================================
# Histogram feature tests
# =============================================================================

class TestHistogramFeatures:
    """Test histogram feature implementations."""
    
    def test_histogram_features_adaptive(self, multi_channel_window):
        """Test histogram features with adaptive range."""
        cpu_result = extract_histogram_features(multi_channel_window, n_bins=10, adaptive_range=True)
        gpu_result = cp.asnumpy(extract_histogram_features_gpu(
            cp.asarray(multi_channel_window), n_bins=10, adaptive_range=True
        ))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_histogram_features_fixed(self, multi_channel_window):
        """Test histogram features with fixed range."""
        cpu_result = extract_histogram_features(multi_channel_window, n_bins=10, adaptive_range=False)
        gpu_result = cp.asnumpy(extract_histogram_features_gpu(
            cp.asarray(multi_channel_window), n_bins=10, adaptive_range=False
        ))
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)


# =============================================================================
# Combined feature tests
# =============================================================================

class TestCombinedFeatures:
    """Test combined feature extraction."""
    
    def test_combined_features_default(self, multi_channel_window):
        """Test combined features with default options."""
        cpu_result = extract_combined_features(multi_channel_window, sampling_rate=200.0)
        gpu_result = cp.asnumpy(extract_combined_features_gpu(
            cp.asarray(multi_channel_window), sampling_rate=200.0
        ))
        
        assert cpu_result.shape == gpu_result.shape, \
            f"Shape mismatch: CPU {cpu_result.shape} vs GPU {gpu_result.shape}"
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)
    
    def test_combined_features_with_histogram(self, multi_channel_window):
        """Test combined features with histogram enabled."""
        cpu_result = extract_combined_features(
            multi_channel_window, sampling_rate=200.0, include_histogram=True, hist_bins=10
        )
        gpu_result = cp.asnumpy(extract_combined_features_gpu(
            cp.asarray(multi_channel_window), sampling_rate=200.0, include_histogram=True, hist_bins=10
        ))
        
        assert cpu_result.shape == gpu_result.shape
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)
    
    def test_combined_features_all_time(self, multi_channel_window):
        """Test combined features with all time-domain features enabled."""
        cpu_result = extract_combined_features(
            multi_channel_window, sampling_rate=200.0,
            time_mav=True, time_wl=True, time_zc=True, time_ssc=True,
            time_rms=True, time_iemg=True, time_var=True, time_lscale=True,
        )
        gpu_result = cp.asnumpy(extract_combined_features_gpu(
            cp.asarray(multi_channel_window), sampling_rate=200.0,
            time_mav=True, time_wl=True, time_zc=True, time_ssc=True,
            time_rms=True, time_iemg=True, time_var=True, time_lscale=True,
        ))
        
        assert cpu_result.shape == gpu_result.shape
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)
    
    def test_features_from_config(self, multi_channel_window, feature_config):
        """Test feature extraction using FeatureSetConfig."""
        cpu_result = extract_features_from_config(
            multi_channel_window, feature_config, sampling_rate=200.0
        )
        gpu_result = cp.asnumpy(extract_features_from_config_gpu(
            cp.asarray(multi_channel_window), feature_config, sampling_rate=200.0
        ))
        
        assert cpu_result.shape == gpu_result.shape, \
            f"Shape mismatch: CPU {cpu_result.shape} vs GPU {gpu_result.shape}"
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)


# =============================================================================
# Batch processing tests
# =============================================================================

class TestBatchProcessing:
    """Test batch GPU processing."""
    
    def test_batch_extraction(self, batch_windows):
        """Test batch feature extraction matches individual extraction."""
        # GPU batch extraction
        gpu_batch_result = extract_features_batch_gpu(batch_windows, sampling_rate=200.0)
        
        # CPU individual extraction for comparison
        cpu_results = np.array([
            extract_combined_features(w, sampling_rate=200.0) 
            for w in batch_windows
        ])
        
        assert gpu_batch_result.shape == cpu_results.shape, \
            f"Shape mismatch: GPU batch {gpu_batch_result.shape} vs CPU {cpu_results.shape}"
        np.testing.assert_allclose(gpu_batch_result, cpu_results, rtol=1e-4, atol=1e-3)
    
    def test_batch_with_config(self, batch_windows, feature_config):
        """Test batch extraction with feature config."""
        # GPU batch extraction with config
        gpu_batch_result = extract_features_batch_gpu(
            batch_windows, sampling_rate=200.0, config=feature_config
        )
        
        # CPU individual extraction for comparison
        cpu_results = np.array([
            extract_features_from_config(w, feature_config, sampling_rate=200.0)
            for w in batch_windows
        ])
        
        assert gpu_batch_result.shape == cpu_results.shape
        np.testing.assert_allclose(gpu_batch_result, cpu_results, rtol=1e-4, atol=1e-3)
    
    def test_single_window_gpu(self, multi_channel_window):
        """Test single window GPU extraction."""
        gpu_result = extract_features_gpu(multi_channel_window, sampling_rate=200.0)
        cpu_result = extract_combined_features(multi_channel_window, sampling_rate=200.0)
        
        assert gpu_result.shape == cpu_result.shape
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-4, atol=1e-3)


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_single_channel_time_features(self, single_channel_window):
        """Test time features with single channel input."""
        cpu_result = extract_time_features(single_channel_window)
        gpu_result = cp.asnumpy(extract_time_features_gpu(cp.asarray(single_channel_window)))
        
        assert cpu_result.shape == gpu_result.shape
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)
    
    def test_single_channel_spectral_no_correlation(self, single_channel_window):
        """Test spectral features with single channel (no correlation)."""
        # Single channel correlation is undefined, so disable it
        cpu_result = extract_spectral_features(
            single_channel_window, sampling_rate=200.0, include_correlation=False
        )
        gpu_result = cp.asnumpy(extract_spectral_features_gpu(
            cp.asarray(single_channel_window), sampling_rate=200.0, include_correlation=False
        ))
        
        assert cpu_result.shape == gpu_result.shape
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-3)
    
    def test_different_window_sizes(self):
        """Test with different window sizes."""
        np.random.seed(42)
        for window_size in [50, 100, 200, 400, 512]:
            window = np.random.randn(window_size, 4).astype(np.float32)
            
            cpu_result = extract_combined_features(window, sampling_rate=200.0)
            gpu_result = cp.asnumpy(extract_combined_features_gpu(
                cp.asarray(window), sampling_rate=200.0
            ))
            
            assert cpu_result.shape == gpu_result.shape, \
                f"Shape mismatch for window_size={window_size}"
            np.testing.assert_allclose(
                cpu_result, gpu_result, rtol=1e-4, atol=1e-3,
                err_msg=f"Values mismatch for window_size={window_size}"
            )
    
    def test_different_sampling_rates(self, multi_channel_window):
        """Test with different sampling rates."""
        for sr in [100.0, 200.0, 500.0, 1000.0, 2000.0]:
            cpu_result = extract_combined_features(multi_channel_window, sampling_rate=sr)
            gpu_result = cp.asnumpy(extract_combined_features_gpu(
                cp.asarray(multi_channel_window), sampling_rate=sr
            ))
            
            assert cpu_result.shape == gpu_result.shape
            np.testing.assert_allclose(
                cpu_result, gpu_result, rtol=1e-4, atol=1e-3,
                err_msg=f"Values mismatch for sampling_rate={sr}"
            )
    
    def test_small_window(self):
        """Test with very small window (edge case for SSC)."""
        np.random.seed(42)
        window = np.random.randn(10, 4).astype(np.float32)
        
        cpu_result = extract_time_features(window)
        gpu_result = cp.asnumpy(extract_time_features_gpu(cp.asarray(window)))
        
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)


# =============================================================================
# Feature count consistency tests
# =============================================================================

class TestFeatureCounts:
    """Test that CPU and GPU produce same number of features."""
    
    def test_feature_count_default(self, multi_channel_window):
        """Test feature count with default options."""
        cpu_result = extract_combined_features(multi_channel_window, sampling_rate=200.0)
        gpu_result = cp.asnumpy(extract_combined_features_gpu(
            cp.asarray(multi_channel_window), sampling_rate=200.0
        ))
        
        assert len(cpu_result) == len(gpu_result), \
            f"Feature count mismatch: CPU {len(cpu_result)} vs GPU {len(gpu_result)}"
    
    def test_feature_count_with_config(self, multi_channel_window, feature_config):
        """Test feature count matches when using config."""
        cpu_result = extract_features_from_config(
            multi_channel_window, feature_config, sampling_rate=200.0
        )
        gpu_result = cp.asnumpy(extract_features_from_config_gpu(
            cp.asarray(multi_channel_window), feature_config, sampling_rate=200.0
        ))
        
        assert len(cpu_result) == len(gpu_result), \
            f"Feature count mismatch: CPU {len(cpu_result)} vs GPU {len(gpu_result)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
