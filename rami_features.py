import numpy as np


def compute_fft(x, fs=1000):
    """Compute magnitude spectrum for each channel.
    x : [winsize × num_channels]
    """
    winsize = x.shape[0]
    freqs = np.fft.rfftfreq(winsize, 1 / fs)
    X = np.abs(np.fft.rfft(x, axis=0))
    return freqs, X


def spectral_moments(freqs, X):
    """Compute M0–M4 for each channel."""
    eps = 1e-12
    moments = []

    for k in range(5):
        num = np.sum((freqs[:, None] ** k) * X, axis=0)
        den = np.sum(X, axis=0) + eps
        moments.append(num / den)

    return np.concatenate(moments, axis=0)  # 5 * num_channels


def spectral_flux(X):
    """Flux for each channel."""
    diff = np.diff(X, axis=0)
    flux = np.sum(diff ** 2, axis=0)
    return flux


def spectral_sparsity(X):
    """Sparsity for each channel."""
    N = X.shape[0]
    l1 = np.sum(np.abs(X), axis=0)
    l2 = np.sqrt(np.sum(X ** 2, axis=0))
    return (np.sqrt(N) * l1) / (l2 + 1e-12)


def irregularity_factor(X):
    """Irregularity factor for each channel."""
    diff = np.diff(X, axis=0)
    return np.sum(diff ** 2, axis=0)


def spectrum_correlation(X, reference=None):
    """Power spectrum correlation.

    If reference is None → compute correlation between channels.
    """
    P = X ** 2  # power spectrum

    if reference is not None:
        # correlation with reference spectrum
        corr = []
        for ch in range(X.shape[1]):
            corr.append(np.corrcoef(P[:, ch], reference)[0, 1])
        return np.array(corr)

    # otherwise compute inter-channel correlation vector
    C = np.corrcoef(P.T)
    # return upper triangle (excluding diagonal)
    return C[np.triu_indices_from(C, k=1)]


def extract_khushaba_features(window, fs=1000, reference_spectrum=None):
    """
    Full feature vector for a single EMG window (winsize × channels).
    """
    freqs, X = compute_fft(window, fs)

    feats = []

    # spectral moments: M0..M4 (5 * channels)
    feats.append(spectral_moments(freqs, X))

    # flux (1 * channels)
    feats.append(spectral_flux(X))

    # sparsity (1 * channels)
    feats.append(spectral_sparsity(X))

    # irregularity (1 * channels)
    feats.append(irregularity_factor(X))

    # correlation features
    corr = spectrum_correlation(X, reference_spectrum)
    feats.append(corr)

    return np.concatenate(feats)
