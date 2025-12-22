"""DB1 (NinaPro Database 1) loader for EMG data.

Loads EMG data from NinaPro DB1 MATLAB .mat files, handling the specific
format with continuous EMG streams, per-sample labels, and exercise offsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np
from scipy.io import loadmat

from ..base import DataLoader, EMGData


# NinaPro DB1 exercise label offsets for globally unique labels
# E1: 12 movements (1-12), E2: 17 movements (13-29), E3: 23 movements (30-52)
# Rest (0) stays as 0 across all exercises
DB1_LABEL_OFFSETS = {
    1: 0,   # E1: labels 1-12 stay as 1-12
    2: 12,  # E2: labels 1-17 become 13-29
    3: 29,  # E3: labels 1-23 become 30-52
}


class DB1Loader(DataLoader):
    """Load EMG data from NinaPro Database 1.
    
    NinaPro DB1 specifics:
    - 27 intact subjects
    - 52 movements across 3 exercises
    - 10 EMG channels (8 forearm + 2 finger extensors)
    - 2 kHz sampling rate (downsampled from 10 kHz)
    - Continuous recordings with per-sample stimulus labels
    - 'restimulus' provides refined timing, 'stimulus' is original
    
    Each .mat file contains a single exercise for one subject with:
    - emg: continuous EMG signal (n_samples, n_channels)
    - stimulus/restimulus: per-sample movement labels
    - repetition/rerepetition: repetition numbers per sample
    - exercise: exercise number (1, 2, or 3)
    - subject: subject number
    """
    
    # Dataset-specific defaults
    DEFAULT_N_CHANNELS = 10
    DEFAULT_SAMPLING_RATE = 2000.0
    DEFAULT_FILE_PATTERN = "**/*.mat"
    DEFAULT_LABEL_COLUMN = "restimulus"
    
    def __init__(
        self,
        data_dir: Union[Path, str],
        n_channels: int = DEFAULT_N_CHANNELS,
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        file_pattern: str = DEFAULT_FILE_PATTERN,
        label_column: str = DEFAULT_LABEL_COLUMN,
        emg_columns: Optional[Tuple[int, int]] = None,
        global_labels: bool = True,
    ):
        """Initialize the DB1 loader.
        
        Args:
            data_dir: Directory containing .mat files (searched recursively).
            n_channels: Number of EMG channels to load (default: 10).
            sampling_rate: Sampling rate in Hz (default: 2000.0).
            file_pattern: Glob pattern for finding .mat files.
            label_column: Column to use for labels ('stimulus' or 'restimulus').
                         'restimulus' is recommended as it has refined timing.
            emg_columns: Tuple of (start, end) column indices for EMG channels.
                        None = use all channels (0-10), (0, 8) = forearm only.
            global_labels: If True, offset labels by exercise to create globally
                          unique labels across exercises (E1: 1-12, E2: 13-29, 
                          E3: 30-52). Rest (0) stays as 0.
        """
        super().__init__(
            data_dir=data_dir,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            file_pattern=file_pattern,
            label_mapping=None,
        )
        self.label_column = label_column
        self.emg_columns = emg_columns
        self.global_labels = global_labels
    
    def load_file(self, path: Path) -> EMGData:
        """Load a single .mat file.
        
        Args:
            path: Path to the .mat file.
            
        Returns:
            EMGData with continuous EMG signal and per-sample labels in metadata.
        """
        # Load MATLAB file
        mat_data = loadmat(str(path), squeeze_me=False)
        
        # Extract EMG data
        emg = mat_data["emg"]
        
        # Select EMG columns if specified
        if self.emg_columns is not None:
            start, end = self.emg_columns
            emg = emg[:, start:end]
        else:
            emg = emg[:, :self.n_channels]
        
        # Extract labels (per-sample)
        labels = mat_data[self.label_column].flatten().astype(int)
        
        # Extract repetition info
        if "rerepetition" in mat_data and self.label_column == "restimulus":
            repetition = mat_data["rerepetition"].flatten()
        else:
            repetition = mat_data.get("repetition", np.zeros(len(labels))).flatten()
        
        # Extract metadata from file and mat contents
        metadata = self._extract_metadata(path, mat_data)
        
        # Apply global label offset if enabled (makes labels unique across exercises)
        if self.global_labels and "exercise" in metadata:
            exercise = metadata["exercise"]
            offset = DB1_LABEL_OFFSETS.get(exercise, 0)
            # Only offset non-rest labels (rest=0 stays as 0)
            labels = np.where(labels > 0, labels + offset, labels)
        
        # Store per-sample labels and repetitions in metadata
        # This allows windowing to assign labels to windows
        metadata["sample_labels"] = labels
        metadata["sample_repetitions"] = repetition
        
        # For compatibility, use the most common non-rest label as the file label
        non_rest_labels = labels[labels > 0]
        if len(non_rest_labels) > 0:
            file_label = int(np.bincount(non_rest_labels.astype(int)).argmax())
        else:
            file_label = 0
        
        return EMGData(emg=emg, label=file_label, metadata=metadata)
    
    def get_label(self, path: Path) -> int:
        """Extract the dominant class label for a .mat file.
        
        For NinaPro files, this returns the most common non-rest label,
        but per-sample labels should be used for windowing.
        
        Args:
            path: Path to the .mat file.
            
        Returns:
            Integer class label (most common non-rest movement).
        """
        mat_data = loadmat(str(path), squeeze_me=False)
        labels = mat_data[self.label_column].flatten()
        
        non_rest_labels = labels[labels > 0]
        if len(non_rest_labels) > 0:
            return int(np.bincount(non_rest_labels.astype(int)).argmax())
        return 0
    
    def _extract_metadata(self, path: Path, mat_data: dict) -> dict:
        """Extract metadata from filename and .mat contents.
        
        Args:
            path: Path to the .mat file.
            mat_data: Loaded MATLAB data dictionary.
            
        Returns:
            Dictionary with extracted metadata.
        """
        metadata = {
            "filename": path.name,
            "filepath": str(path),
        }
        
        # Extract subject from mat file or path
        if "subject" in mat_data:
            metadata["subject"] = f"s{int(mat_data['subject'].flat[0]):02d}"
        else:
            # Try to extract from path (e.g., s01, s02)
            for part in path.parts:
                if re.match(r"s\d+", part, re.IGNORECASE):
                    metadata["subject"] = part.lower()
                    break
            else:
                metadata["subject"] = path.parent.name
        
        # Extract exercise number
        if "exercise" in mat_data:
            exercise = int(mat_data["exercise"].flat[0])
            metadata["exercise"] = exercise
            metadata["position"] = f"E{exercise}"
        else:
            # Try to extract from filename (e.g., S1_A1_E1.mat)
            match = re.search(r"E(\d+)", path.name)
            if match:
                exercise = int(match.group(1))
                metadata["exercise"] = exercise
                metadata["position"] = f"E{exercise}"
            else:
                metadata["position"] = "unknown"
        
        return metadata
    
    def discover_files(self) -> List[Path]:
        """Find all .mat files matching the pattern.
        
        Returns:
            Sorted list of file paths.
        """
        # Use rglob for recursive search with ** pattern
        if "**" in self.file_pattern:
            pattern = self.file_pattern.replace("**/", "")
            files = list(self.data_dir.rglob(pattern))
        else:
            files = list(self.data_dir.glob(self.file_pattern))
        
        return sorted(files)
    
    def load_by_subject(
        self,
        subjects: Optional[List[str]] = None,
    ) -> Dict[str, List[EMGData]]:
        """Load files grouped by subject.
        
        Args:
            subjects: List of subject IDs to include (e.g., ['s01', 's02']).
                     If None, load all subjects.
            
        Returns:
            Dictionary mapping subject ID -> list of EMGData.
        """
        from collections import defaultdict
        
        result: Dict[str, List[EMGData]] = defaultdict(list)
        
        for path in self.discover_files():
            data = self.load_file(path)
            subject = data.metadata.get("subject", "unknown")
            
            if subjects is None or subject in subjects:
                result[subject].append(data)
        
        return dict(result)
    
    def load_by_exercise(
        self,
        exercises: Optional[List[int]] = None,
    ) -> Dict[int, List[EMGData]]:
        """Load files grouped by exercise.
        
        Args:
            exercises: List of exercise numbers to include (e.g., [1, 2, 3]).
                      If None, load all exercises.
            
        Returns:
            Dictionary mapping exercise number -> list of EMGData.
        """
        from collections import defaultdict
        
        result: Dict[int, List[EMGData]] = defaultdict(list)
        
        for path in self.discover_files():
            data = self.load_file(path)
            exercise = data.metadata.get("exercise", 0)
            
            if exercises is None or exercise in exercises:
                result[exercise].append(data)
        
        return dict(result)
