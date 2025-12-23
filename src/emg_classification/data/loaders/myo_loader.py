"""GrabMyo dataset loader for EMG data.

Loads EMG data from WFDB format files used in the GrabMyo dataset
from PhysioNet. Supports channel group selection and multi-session handling.

Reference: https://physionet.org/content/grabmyo/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import wfdb
except ImportError:
    wfdb = None

from ..base import DataLoader, EMGData


class MyoLoader(DataLoader):
    """Load EMG data from the GrabMyo dataset (WFDB format).
    
    GrabMyo dataset specifics:
    - 43 subjects
    - 3 sessions per subject
    - 17 gestures (16 movements + rest)
    - 28 EMG channels total:
      - Forearm band: channels 0-15 (16 channels)
      - Wrist band: channels 16-27 (12 channels)
    - 2048 Hz sampling rate
    - WFDB format (.dat + .hea files)
    
    Filename pattern: session{i}_participant{j}_gesture{k}_trial{l}
    
    Directory structure:
        data_dir/
            Session1/
                session1_participant1/
                    session1_participant1_gesture1_trial1.dat
                    session1_participant1_gesture1_trial1.hea
                    ...
            Session2/
                ...
            Session3/
                ...
    
    Example:
        >>> loader = MyoLoader("datasets/myo/data", channel_group="forearm", sessions=[1, 2])
        >>> data = loader.load_all()  # Load all forearm data from sessions 1 & 2
        >>> data[0].emg.shape  # (n_samples, 16)
    """
    
    # Dataset-specific defaults
    DEFAULT_N_CHANNELS = 28
    DEFAULT_SAMPLING_RATE = 2048.0
    DEFAULT_FILE_PATTERN = "**/*.dat"
    
    # Channel group definitions
    CHANNEL_GROUPS = {
        "forearm": (0, 16),   # Forearm band: channels 0-15
        "wrist": (16, 28),    # Wrist band: channels 16-27
        "all": (0, 28),       # All channels
    }
    
    # Filename parsing pattern
    FILENAME_PATTERN = re.compile(
        r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)"
    )
    
    def __init__(
        self,
        data_dir: Union[Path, str],
        n_channels: int = DEFAULT_N_CHANNELS,
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        file_pattern: str = DEFAULT_FILE_PATTERN,
        channel_group: str = "all",
        sessions: Optional[List[int]] = None,
    ):
        """Initialize the GrabMyo loader.
        
        Args:
            data_dir: Directory containing WFDB files (searched recursively).
            n_channels: Number of EMG channels (auto-set based on channel_group).
            sampling_rate: Sampling rate in Hz (default: 2048.0).
            file_pattern: Glob pattern for finding .dat files.
            channel_group: Which electrode group to use:
                - "forearm": channels 0-15 (16 channels)
                - "wrist": channels 16-27 (12 channels)
                - "all": all 28 channels
            sessions: List of sessions to include (e.g., [1, 2, 3]).
                     If None, include all sessions.
        
        Raises:
            ValueError: If channel_group is not valid.
            ImportError: If wfdb package is not installed.
        """
        if wfdb is None:
            raise ImportError(
                "wfdb package is required for MyoLoader. "
                "Install it with: pip install wfdb"
            )
        
        if channel_group not in self.CHANNEL_GROUPS:
            raise ValueError(
                f"Invalid channel_group: {channel_group}. "
                f"Must be one of: {list(self.CHANNEL_GROUPS.keys())}"
            )
        
        # Set n_channels based on channel_group
        start, end = self.CHANNEL_GROUPS[channel_group]
        n_channels = end - start
        
        super().__init__(
            data_dir=data_dir,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
            file_pattern=file_pattern,
            label_mapping=None,
        )
        self.channel_group = channel_group
        self.sessions = sessions
        self._channel_start, self._channel_end = start, end
    
    def load_file(self, path: Path) -> EMGData:
        """Load a single WFDB file.
        
        Args:
            path: Path to the .dat file (or record name without extension).
            
        Returns:
            EMGData with the loaded signal and metadata.
            
        Raises:
            FileNotFoundError: If the WFDB record files don't exist.
            ValueError: If the file cannot be parsed or data is invalid.
        """
        # wfdb.rdrecord expects path without extension
        record_path = path.parent / path.stem
        
        # Load the WFDB record
        record = wfdb.rdrecord(str(record_path))
        
        # Extract the signal (shape: n_samples, n_channels)
        emg = record.p_signal
        
        # Select channel group
        emg = emg[:, self._channel_start:self._channel_end]
        
        # Ensure float32 for memory efficiency
        emg = emg.astype(np.float32)
        
        # Extract metadata from filename
        metadata = self._extract_metadata(path)
        
        # Add WFDB record info to metadata
        metadata["sampling_rate"] = record.fs
        metadata["n_samples"] = record.sig_len
        metadata["channel_group"] = self.channel_group
        
        # Label is the gesture number (1-indexed in dataset)
        label = metadata.get("gesture", 0)
        
        return EMGData(emg=emg, label=label, metadata=metadata)
    
    def get_label(self, path: Path) -> int:
        """Extract class label from filename.
        
        Filename pattern: session{i}_participant{j}_gesture{k}_trial{l}
        
        Args:
            path: Path to the data file.
            
        Returns:
            Integer class label (gesture number, 1-17).
        """
        match = self.FILENAME_PATTERN.search(path.stem)
        if match:
            return int(match.group(3))  # gesture number
        return 0
    
    def _extract_metadata(self, path: Path) -> dict:
        """Extract metadata from filename.
        
        Filename pattern: session{i}_participant{j}_gesture{k}_trial{l}
        
        Args:
            path: Path to the data file.
            
        Returns:
            Dictionary with extracted metadata including:
            - filename: Original filename
            - filepath: Full path as string
            - session: Session number (1-3)
            - subject: Subject ID formatted as 's{num:02d}'
            - participant: Raw participant number
            - gesture: Gesture number (1-17)
            - trial: Trial number (1-7)
            - repetition: Same as trial (for compatibility)
            - position: Gesture formatted as 'G{num}'
        """
        name = path.stem
        metadata = {
            "filename": path.name,
            "filepath": str(path),
        }
        
        # Parse filename using compiled pattern
        match = self.FILENAME_PATTERN.search(name)
        if match:
            session = int(match.group(1))
            participant = int(match.group(2))
            gesture = int(match.group(3))
            trial = int(match.group(4))
            
            metadata["session"] = session
            metadata["participant"] = participant
            metadata["subject"] = f"s{participant:02d}"
            metadata["gesture"] = gesture
            metadata["trial"] = trial
            metadata["repetition"] = trial  # Alias for compatibility
            metadata["position"] = f"G{gesture}"  # Alias for compatibility
        
        return metadata
    
    def discover_files(self) -> List[Path]:
        """Find all WFDB .dat files matching the pattern.
        
        Searches recursively through Session1/, Session2/, Session3/ directories.
        
        Returns:
            Sorted list of file paths, filtered by session if specified.
        """
        # Use rglob for recursive search with ** pattern
        if "**" in self.file_pattern:
            pattern = self.file_pattern.replace("**/", "")
            files = list(self.data_dir.rglob(pattern))
        else:
            files = list(self.data_dir.glob(self.file_pattern))
        
        # Filter by session if specified
        if self.sessions is not None:
            filtered = []
            for f in files:
                match = self.FILENAME_PATTERN.search(f.stem)
                if match and int(match.group(1)) in self.sessions:
                    filtered.append(f)
            files = filtered
        
        return sorted(files)
    
    def load_by_subject(
        self,
        subjects: Optional[List[str]] = None,
    ) -> Dict[str, List[EMGData]]:
        """Load files grouped by subject (participant).
        
        Args:
            subjects: List of subject IDs to include (e.g., ['s01', 's02']).
                     If None, load all subjects.
            
        Returns:
            Dictionary mapping subject ID -> list of EMGData.
        """
        files = self.discover_files()
        result: Dict[str, List[EMGData]] = {}
        
        for f in files:
            match = self.FILENAME_PATTERN.search(f.stem)
            if not match:
                continue
            
            participant = int(match.group(2))
            subject_id = f"s{participant:02d}"
            
            # Filter by subject if specified
            if subjects is not None and subject_id not in subjects:
                continue
            
            if subject_id not in result:
                result[subject_id] = []
            
            result[subject_id].append(self.load_file(f))
        
        return result
    
    def load_by_session(
        self,
        sessions: Optional[List[int]] = None,
    ) -> Dict[int, List[EMGData]]:
        """Load files grouped by session.
        
        Args:
            sessions: List of session numbers to include (e.g., [1, 2, 3]).
                     If None, load all sessions.
            
        Returns:
            Dictionary mapping session number -> list of EMGData.
        """
        files = self.discover_files()
        result: Dict[int, List[EMGData]] = {}
        
        for f in files:
            match = self.FILENAME_PATTERN.search(f.stem)
            if not match:
                continue
            
            session = int(match.group(1))
            
            # Filter by session if specified (in addition to self.sessions filter)
            if sessions is not None and session not in sessions:
                continue
            
            if session not in result:
                result[session] = []
            
            result[session].append(self.load_file(f))
        
        return result
    
    def load_by_subject_and_session(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[int]] = None,
    ) -> Dict[str, Dict[int, List[EMGData]]]:
        """Load files grouped by subject, then by session.
        
        Useful for cross-session evaluation within subjects.
        
        Args:
            subjects: List of subject IDs to include (e.g., ['s01', 's02']).
                     If None, load all subjects.
            sessions: List of session numbers to include (e.g., [1, 2, 3]).
                     If None, load all sessions.
            
        Returns:
            Nested dictionary: subject_id -> session_number -> list of EMGData.
        """
        files = self.discover_files()
        result: Dict[str, Dict[int, List[EMGData]]] = {}
        
        for f in files:
            match = self.FILENAME_PATTERN.search(f.stem)
            if not match:
                continue
            
            session = int(match.group(1))
            participant = int(match.group(2))
            subject_id = f"s{participant:02d}"
            
            # Filter
            if subjects is not None and subject_id not in subjects:
                continue
            if sessions is not None and session not in sessions:
                continue
            
            if subject_id not in result:
                result[subject_id] = {}
            if session not in result[subject_id]:
                result[subject_id][session] = []
            
            result[subject_id][session].append(self.load_file(f))
        
        return result
    
    def get_subject_ids(self) -> List[str]:
        """Get all unique subject IDs in the dataset.
        
        Returns:
            Sorted list of subject IDs (e.g., ['s01', 's02', ...]).
        """
        files = self.discover_files()
        subjects = set()
        
        for f in files:
            match = self.FILENAME_PATTERN.search(f.stem)
            if match:
                participant = int(match.group(2))
                subjects.add(f"s{participant:02d}")
        
        return sorted(subjects)
    
    def get_session_ids(self) -> List[int]:
        """Get all unique session IDs in the dataset.
        
        Returns:
            Sorted list of session numbers (e.g., [1, 2, 3]).
        """
        files = self.discover_files()
        sessions = set()
        
        for f in files:
            match = self.FILENAME_PATTERN.search(f.stem)
            if match:
                sessions.add(int(match.group(1)))
        
        return sorted(sessions)
