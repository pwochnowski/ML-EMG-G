"""Signal preprocessing filters for EMG signals.

Applies bandpass and notch filtering to raw EMG signals before
windowing and feature extraction.

Based on standard EMG preprocessing practices:
- Bandpass 20-500 Hz to retain relevant EMG frequency content
- Notch filter at 50/60 Hz to remove powerline interference
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy import signal

from ..config.schema import PreprocessingConfig


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to EMG signal.
    
    Args:
        data: EMG signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz. Will be clamped to Nyquist if needed.
        order: Filter order.
        
    Returns:
        Filtered signal with same shape as input.
    """
    nyquist = fs / 2.0
    
    # Clamp highcut to just below Nyquist
    highcut = min(highcut, nyquist * 0.99)
    
    # Ensure lowcut < highcut
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be less than highcut ({highcut})")
    
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering (filtfilt applies filter twice, forward and backward)
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Filter each channel independently
        return signal.filtfilt(b, a, data, axis=0)


def notch_filter(
    data: np.ndarray,
    fs: float,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """Apply notch filter to remove powerline interference.
    
    Args:
        data: EMG signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        freq: Notch frequency in Hz (50 Hz for EU, 60 Hz for US).
        q: Quality factor. Higher values = narrower notch.
        
    Returns:
        Filtered signal with same shape as input.
    """
    nyquist = fs / 2.0
    
    # Skip if notch frequency is above Nyquist
    if freq >= nyquist:
        return data
    
    # Design notch filter
    b, a = signal.iirnotch(freq, q, fs)
    
    # Apply zero-phase filtering
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return signal.filtfilt(b, a, data, axis=0)


class SignalPreprocessor:
    """Preprocessor for raw EMG signals.
    
    Applies configurable filtering to raw EMG signals before windowing
    and feature extraction.
    
    Example:
        >>> from emg_classification.preprocessing import SignalPreprocessor
        >>> from emg_classification.config.schema import load_preprocessing_config
        >>> 
        >>> config = load_preprocessing_config()
        >>> preprocessor = SignalPreprocessor(config)
        >>> filtered_signal = preprocessor.process(raw_emg, fs=2000.0)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration. If None, uses default settings.
        """
        self.config = config or PreprocessingConfig()
    
    def process(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply all configured preprocessing steps to the signal.
        
        Processing order:
        1. Bandpass filter (if enabled)
        2. Notch filter (if enabled)
        
        Args:
            data: Raw EMG signal of shape (n_samples,) or (n_samples, n_channels).
            fs: Sampling frequency in Hz.
            
        Returns:
            Preprocessed signal with same shape as input.
        """
        result = data.copy()
        
        # Apply bandpass filter
        if self.config.bandpass_enabled:
            result = bandpass_filter(
                result,
                fs=fs,
                lowcut=self.config.bandpass_lowcut,
                highcut=self.config.bandpass_highcut,
                order=self.config.bandpass_order,
            )
        
        # Apply notch filter
        if self.config.notch_enabled:
            result = notch_filter(
                result,
                fs=fs,
                freq=self.config.notch_freq,
                q=self.config.notch_q,
            )
        
        return result
    
    def __repr__(self) -> str:
        parts = []
        if self.config.bandpass_enabled:
            parts.append(f"bandpass({self.config.bandpass_lowcut}-{self.config.bandpass_highcut}Hz)")
        if self.config.notch_enabled:
            parts.append(f"notch({self.config.notch_freq}Hz)")
        return f"SignalPreprocessor([{', '.join(parts)}])"
