"""Windowing utilities for EMG signals."""

import numpy as np


def sliding_window(
    x: np.ndarray,
    window_size: int,
    window_increment: int,
) -> np.ndarray:
    """Apply sliding window to a signal.
    
    Args:
        x: Input signal of shape (n_samples,) or (n_samples, n_channels).
        window_size: Size of each window in samples.
        window_increment: Step size between windows in samples.
        
    Returns:
        Array of windows with shape (n_windows, window_size, n_channels).
        Returns empty array of shape (0, window_size, n_channels) if
        the signal is shorter than window_size.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    
    n_samples, n_channels = x.shape
    
    if n_samples < window_size:
        return np.zeros((0, window_size, n_channels), dtype=x.dtype)
    
    # Calculate number of windows
    n_windows = (n_samples - window_size) // window_increment + 1
    
    # Create window indices
    indices = np.arange(n_windows) * window_increment
    
    # Extract windows
    windows = np.stack([x[i:i + window_size] for i in indices])
    
    return windows


def apply_window_function(
    windows: np.ndarray,
    window_type: str = "hamming",
) -> np.ndarray:
    """Apply a window function to windowed data.
    
    Args:
        windows: Array of shape (n_windows, window_size, n_channels).
        window_type: Type of window function ('hamming', 'hann', 'blackman', etc.).
        
    Returns:
        Windowed data with the window function applied.
    """
    window_size = windows.shape[1]
    
    if window_type == "hamming":
        window = np.hamming(window_size)
    elif window_type == "hann":
        window = np.hanning(window_size)
    elif window_type == "blackman":
        window = np.blackman(window_size)
    elif window_type == "rectangular" or window_type == "none":
        return windows
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Apply window to each channel
    return windows * window[:, np.newaxis]
