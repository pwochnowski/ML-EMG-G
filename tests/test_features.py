"""Tests for feature extraction."""

import numpy as np
import pytest

from emg_classification.features import (
    extract_spectral_features,
    extract_time_features,
    extract_combined_features,
)


@pytest.fixture
def sample_window():
    """Create a sample EMG window for testing."""
    np.random.seed(42)
    # 400 samples, 7 channels
    return np.random.randn(400, 7)


class TestSpectralFeatures:
    def test_output_shape(self, sample_window):
        """Test that spectral features have correct shape."""
        features = extract_spectral_features(sample_window)
        n_channels = sample_window.shape[1]
        # 5 moments + 1 flux + 1 sparsity + 1 irregularity per channel
        # + n_channels*(n_channels-1)/2 correlations
        expected_per_channel = 5 + 1 + 1 + 1
        expected_corr = n_channels * (n_channels - 1) // 2
        expected_total = expected_per_channel * n_channels + expected_corr
        assert features.shape == (expected_total,)
    
    def test_single_channel(self):
        """Test with single channel input."""
        window = np.random.randn(400, 1)
        features = extract_spectral_features(window)
        # 5 moments + flux + sparsity + irregularity + 0 correlations
        assert features.shape == (8,)
    
    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        window = np.random.randn(400)
        features = extract_spectral_features(window)
        assert features.ndim == 1


class TestTimeFeatures:
    def test_output_shape(self, sample_window):
        """Test that time features have correct shape."""
        features = extract_time_features(sample_window)
        n_channels = sample_window.shape[1]
        # 4 features (MAV, WL, ZC, SSC) per channel
        assert features.shape == (4 * n_channels,)
    
    def test_single_channel(self):
        """Test with single channel input."""
        window = np.random.randn(400, 1)
        features = extract_time_features(window)
        assert features.shape == (4,)
    
    def test_mav_non_negative(self, sample_window):
        """Test that MAV is non-negative."""
        features = extract_time_features(sample_window, include_wl=False, include_zc=False, include_ssc=False)
        assert np.all(features >= 0)


class TestCombinedFeatures:
    def test_output_combines_both(self, sample_window):
        """Test that combined features include both spectral and time."""
        spectral = extract_spectral_features(sample_window)
        time = extract_time_features(sample_window)
        combined = extract_combined_features(sample_window)
        assert combined.shape[0] == spectral.shape[0] + time.shape[0]
