"""Base classes and protocols for data loading.

Defines the abstract DataLoader interface and EMGData container.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np


@dataclass
class EMGData:
    """Container for EMG data from a single recording.
    
    Attributes:
        emg: Raw EMG signal array of shape (n_samples, n_channels).
        label: Class label for this recording.
        metadata: Additional information (subject, position, repetition, etc.).
    """
    emg: np.ndarray
    label: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return self.emg.shape[0]
    
    @property
    def n_channels(self) -> int:
        """Number of EMG channels."""
        return self.emg.shape[1] if self.emg.ndim > 1 else 1
    
    def __repr__(self) -> str:
        return (
            f"EMGData(shape={self.emg.shape}, label={self.label}, "
            f"metadata={list(self.metadata.keys())})"
        )


class DataLoader(ABC):
    """Abstract base class for EMG data loaders.
    
    Subclasses implement loading logic for specific file formats
    (text files, .mat files, CSV, etc.).
    """
    
    def __init__(
        self,
        data_dir: Union[Path, str],
        n_channels: int,
        sampling_rate: float,
        file_pattern: str = "*",
        label_mapping: Optional[Dict[str, int]] = None,
    ):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files.
            n_channels: Expected number of EMG channels.
            sampling_rate: Sampling rate in Hz.
            file_pattern: Glob pattern for finding data files.
            label_mapping: Optional mapping from identifiers to class labels.
        """
        self.data_dir = Path(data_dir)
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.file_pattern = file_pattern
        self.label_mapping = label_mapping or {}
    
    @abstractmethod
    def load_file(self, path: Path) -> EMGData:
        """Load a single data file.
        
        Args:
            path: Path to the data file.
            
        Returns:
            EMGData instance containing the loaded data.
        """
        pass
    
    @abstractmethod
    def get_label(self, path: Path) -> int:
        """Extract the class label for a data file.
        
        Args:
            path: Path to the data file.
            
        Returns:
            Integer class label.
        """
        pass
    
    def discover_files(self) -> List[Path]:
        """Find all data files matching the pattern.
        
        Returns:
            Sorted list of file paths.
        """
        return sorted(self.data_dir.glob(self.file_pattern))
    
    def load_all(self) -> List[EMGData]:
        """Load all data files in the directory.
        
        Returns:
            List of EMGData instances.
        """
        files = self.discover_files()
        return [self.load_file(f) for f in files]
    
    def __iter__(self) -> Iterator[EMGData]:
        """Iterate over all data files."""
        for path in self.discover_files():
            yield self.load_file(path)
    
    def __len__(self) -> int:
        """Number of data files."""
        return len(self.discover_files())


@dataclass
class WindowedData:
    """Container for windowed EMG features.
    
    Attributes:
        X: Feature matrix of shape (n_windows, n_features).
        y: Label array of shape (n_windows,).
        groups: Subject/group identifiers for each window.
        positions: Position identifiers for each window.
        source_files: Source file names for each window.
    """
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    positions: np.ndarray
    source_files: np.ndarray
    
    @property
    def n_samples(self) -> int:
        """Number of windows/samples."""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features per window."""
        return self.X.shape[1]
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return len(np.unique(self.y))
    
    def save(self, path: Union[Path, str]) -> None:
        """Save windowed data to compressed numpy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            groups=self.groups,
            positions=self.positions,
            files=self.source_files,
        )
    
    @classmethod
    def load(cls, path: Union[Path, str]) -> WindowedData:
        """Load windowed data from numpy file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            X=data["X"],
            y=data["y"],
            groups=data.get("groups", np.array([])),
            positions=data.get("positions", np.array([])),
            source_files=data.get("files", np.array([])),
        )
    
    def __repr__(self) -> str:
        return (
            f"WindowedData(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"n_classes={self.n_classes})"
        )
