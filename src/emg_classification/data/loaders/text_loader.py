"""Text file loader for EMG data.

Loads EMG data from whitespace-separated text files, supporting the
format used in the Rami dataset and similar text-based EMG recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np

from ..base import DataLoader, EMGData


# Default label mapping for the Rami dataset
DEFAULT_RAMI_LABEL_MAPPING = {
    "WristFlex": 1,
    "WristExte": 2,
    "WristPron": 3,
    "WristSupi": 4,
    "ObjectGri": 5,
    "PichGrip_": 6,
    "HandOpen_": 7,
    "HandRest_": 8,
}


class TextFileLoader(DataLoader):
    """Load EMG data from whitespace-separated text files.
    
    Expected file format:
    - Each row is a time sample
    - Columns are EMG channels (whitespace-separated)
    - Filename contains position and gesture information
    
    Example filename pattern: Pos1_WristFlexion_Rep1.txt
    """
    
    def __init__(
        self,
        data_dir: Union[Path, str],
        n_channels: int,
        sampling_rate: float = 1000.0,
        file_pattern: str = "Pos*_*.txt",
        label_mapping: Optional[Dict[str, int]] = None,
        position_pattern: str = r"(Pos\d+)_",
        subject_mapping: Optional[Dict[str, str]] = None,
    ):
        """Initialize the text file loader.
        
        Args:
            data_dir: Directory containing text files.
            n_channels: Number of EMG channels to load.
            sampling_rate: Sampling rate in Hz.
            file_pattern: Glob pattern for finding data files.
            label_mapping: Mapping from filename prefix (first 9 chars after position)
                          to class label. Defaults to Rami dataset mapping.
            position_pattern: Regex pattern to extract position from filename.
            subject_mapping: Optional mapping from folder names to standardized
                            subject IDs (e.g., {'S1_Male': 's01'}).
        """
        if label_mapping is None:
            label_mapping = DEFAULT_RAMI_LABEL_MAPPING
        
        super().__init__(
            data_dir=data_dir,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            file_pattern=file_pattern,
            label_mapping=label_mapping,
        )
        self.position_pattern = position_pattern
        self.subject_mapping = subject_mapping or {}
    
    def load_file(self, path: Path) -> EMGData:
        """Load a single text file.
        
        Args:
            path: Path to the text file.
            
        Returns:
            EMGData with the loaded signal and metadata.
        """
        # Load the raw data
        data = np.loadtxt(path)
        
        # Handle 1D data (single channel)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        
        # Truncate to expected number of channels
        data = data[:, :self.n_channels]
        
        # Get label and metadata
        label = self.get_label(path)
        metadata = self._extract_metadata(path)
        
        return EMGData(emg=data, label=label, metadata=metadata)
    
    def get_label(self, path: Path) -> int:
        """Extract class label from filename.
        
        The label is determined by matching the first 9 characters
        after the position prefix against the label mapping.
        
        Args:
            path: Path to the data file.
            
        Returns:
            Integer class label.
            
        Raises:
            ValueError: If the filename doesn't match any known label.
        """
        name = path.name
        
        # Remove position prefix if present
        match = re.match(self.position_pattern, name)
        if match:
            emg_name = name[match.end():]
        else:
            emg_name = name
        
        # Match first 9 characters against label mapping
        prefix = emg_name[:9]
        
        if prefix in self.label_mapping:
            return self.label_mapping[prefix]
        
        # Try matching keys as prefixes
        for key, label in self.label_mapping.items():
            if emg_name.startswith(key):
                return label
        
        raise ValueError(f"Unknown category prefix in: {name}")
    
    def _extract_metadata(self, path: Path) -> dict:
        """Extract metadata from filename and path.
        
        Args:
            path: Path to the data file.
            
        Returns:
            Dictionary with extracted metadata.
        """
        name = path.name
        metadata = {
            "filename": name,
            "filepath": str(path),
        }
        
        # Extract position
        match = re.match(self.position_pattern, name)
        if match:
            metadata["position"] = match.group(1)
            emg_name = name[match.end():]
        else:
            metadata["position"] = "unknown"
            emg_name = name
        
        # Extract subject from parent directory name
        folder_name = path.parent.name
        # Use subject_mapping to standardize subject ID if available
        if self.subject_mapping and folder_name in self.subject_mapping:
            metadata["subject"] = self.subject_mapping[folder_name]
        else:
            metadata["subject"] = folder_name
        
        # Try to extract repetition number
        rep_match = re.search(r"_(\d+)\.txt$", name)
        if rep_match:
            metadata["repetition"] = int(rep_match.group(1))
        
        return metadata
    
    def load_by_position(
        self,
        positions: Optional[List[str]] = None,
        limit_per_position: int = 0,
    ) -> Dict[str, List[EMGData]]:
        """Load files grouped by position.
        
        Args:
            positions: List of positions to include (e.g., ['Pos1', 'Pos2']).
                      If None, load all positions.
            limit_per_position: Maximum files per position (0 = no limit).
            
        Returns:
            Dictionary mapping position -> list of EMGData.
        """
        from collections import defaultdict
        
        files_by_pos: dict[str, list[tuple[Path, str]]] = defaultdict(list)
        
        for path in self.discover_files():
            name = path.name
            match = re.match(self.position_pattern, name)
            if match:
                pos = match.group(1)
                if positions is None or pos in positions:
                    emg_name = name[match.end():]
                    files_by_pos[pos].append((path, emg_name))
        
        result: dict[str, list[EMGData]] = {}
        
        for pos in sorted(files_by_pos.keys()):
            items = files_by_pos[pos]
            if limit_per_position > 0:
                items = items[:limit_per_position]
            
            result[pos] = []
            for path, _ in items:
                try:
                    data = self.load_file(path)
                    result[pos].append(data)
                except ValueError as e:
                    print(f"Skipping {path.name}: {e}")
        
        return result
