"""Normalization and domain adaptation utilities for EMG classification.

These methods help reduce subject variability for better cross-subject performance.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class SubjectNormalizer(BaseEstimator, TransformerMixin):
    """Standard z-score normalization.
    
    Fits global statistics during training, applies z-score during prediction.
    """
    
    def __init__(self):
        self.global_mean_ = None
        self.global_std_ = None
    
    def fit(self, X, y=None):
        self.global_mean_ = np.mean(X, axis=0)
        self.global_std_ = np.std(X, axis=0) + 1e-8
        return self
    
    def transform(self, X):
        return (X - self.global_mean_) / self.global_std_


class PercentileNormalizer(BaseEstimator, TransformerMixin):
    """Normalize using percentiles - more robust to outliers.
    
    Maps features to [0, 1] range using percentiles from training data.
    """
    
    def __init__(self, lower: float = 5, upper: float = 95):
        self.lower = lower
        self.upper = upper
        self.p_low_ = None
        self.p_high_ = None
    
    def fit(self, X, y=None):
        self.p_low_ = np.percentile(X, self.lower, axis=0)
        self.p_high_ = np.percentile(X, self.upper, axis=0)
        return self
    
    def transform(self, X):
        X_norm = (X - self.p_low_) / (self.p_high_ - self.p_low_ + 1e-8)
        return np.clip(X_norm, 0, 1)


class ChannelNormalizer(BaseEstimator, TransformerMixin):
    """Normalize features within each EMG channel independently.
    
    Assumes features are organized as [ch1_f1, ch1_f2, ..., ch2_f1, ch2_f2, ...]
    """
    
    def __init__(self, n_channels: int = 7, features_per_channel: int = 15):
        self.n_channels = n_channels
        self.features_per_channel = features_per_channel
        self.scalers_ = None
    
    def fit(self, X, y=None):
        self.scalers_ = []
        for ch in range(self.n_channels):
            start = ch * self.features_per_channel
            end = start + self.features_per_channel
            scaler = StandardScaler()
            scaler.fit(X[:, start:end])
            self.scalers_.append(scaler)
        return self
    
    def transform(self, X):
        X_out = np.zeros_like(X)
        for ch in range(self.n_channels):
            start = ch * self.features_per_channel
            end = start + self.features_per_channel
            X_out[:, start:end] = self.scalers_[ch].transform(X[:, start:end])
        return X_out


class SubjectAdaptiveScaler(BaseEstimator, TransformerMixin):
    """Adaptive scaling with test-time domain adaptation.
    
    During prediction, re-centers data based on test sample statistics.
    This is a form of unsupervised domain adaptation.
    """
    
    def __init__(self, adaptation_strength: float = 0.5):
        self.adaptation_strength = adaptation_strength
        self.train_mean_ = None
        self.train_std_ = None
    
    def fit(self, X, y=None):
        self.train_mean_ = np.mean(X, axis=0)
        self.train_std_ = np.std(X, axis=0) + 1e-8
        return self
    
    def transform(self, X):
        # Standard normalization first
        X_norm = (X - self.train_mean_) / self.train_std_
        
        # If this looks like test data (different distribution), adapt
        test_mean = np.mean(X, axis=0)
        test_std = np.std(X, axis=0) + 1e-8
        
        # Re-center based on test statistics (partial adaptation)
        X_adapted = X_norm - self.adaptation_strength * (test_mean / test_std)
        
        return X_adapted


class FeatureRatioTransformer(BaseEstimator, TransformerMixin):
    """Transform features into ratios to reduce amplitude sensitivity.
    
    Creates ratio features between pairs of original features, making them
    more robust to absolute amplitude differences between subjects.
    """
    
    def __init__(self, n_ratio_features: int = 50, random_state: int = 42):
        self.n_ratio_features = n_ratio_features
        self.random_state = random_state
        self.feature_pairs_ = None
    
    def fit(self, X, y=None):
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)
        
        pairs = []
        for _ in range(self.n_ratio_features):
            f1 = rng.randint(0, n_features)
            f2 = rng.randint(0, n_features)
            while f2 == f1:
                f2 = rng.randint(0, n_features)
            pairs.append((f1, f2))
        
        self.feature_pairs_ = pairs
        return self
    
    def transform(self, X):
        ratios = []
        for f1, f2 in self.feature_pairs_:
            ratio = X[:, f1] / (X[:, f2] + 1e-8)
            ratios.append(ratio)
        return np.column_stack([X, np.array(ratios).T])


# Normalizer registry
NORMALIZERS = {
    "standard": StandardScaler,
    "percentile": PercentileNormalizer,
    "channel": ChannelNormalizer,
    "adaptive": SubjectAdaptiveScaler,
}


def get_normalizer(
    name: str,
    **kwargs,
) -> BaseEstimator:
    """Get a normalizer by name.
    
    Args:
        name: Normalizer name ('standard', 'percentile', 'channel', 'adaptive').
        **kwargs: Additional arguments for the normalizer.
        
    Returns:
        Normalizer instance.
        
    Raises:
        ValueError: If the normalizer name is unknown.
    """
    if name not in NORMALIZERS:
        raise ValueError(
            f"Unknown normalizer: {name}. Available: {list(NORMALIZERS.keys())}"
        )
    return NORMALIZERS[name](**kwargs)


def list_available_normalizers() -> list[str]:
    """List all available normalizer names."""
    return list(NORMALIZERS.keys())
