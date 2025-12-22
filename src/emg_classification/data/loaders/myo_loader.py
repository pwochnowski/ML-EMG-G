"""GrabMyo dataset loader for EMG data.

Loads EMG data from WFDB format files used in the GrabMyo dataset
from PhysioNet. Supports channel group selection and multi-session handling.

Reference: https://physionet.org/content/grabmyo/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

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
    
    TODO: Implement WFDB reading via wfdb.rdrecord()
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
        """
        # Set n_channels based on channel_group
        if channel_group in self.CHANNEL_GROUPS:
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
    
    def load_file(self, path: Path) -> EMGData:
        """Load a single WFDB file.
        
        Args:
            path: Path to the .dat file (or record name without extension).
            
        Returns:
            EMGData with the loaded signal and metadata.
            
        Raises:
            NotImplementedError: This is a stub - WFDB loading not yet implemented.
        """
        raise NotImplementedError(
            "MyoLoader.load_file() is not yet implemented. "
            "Requires wfdb package: pip install wfdb"
        )
        
        # TODO: Implementation outline:
        # import wfdb
        # record_name = path.stem  # Remove .dat extension
        # record = wfdb.rdrecord(str(path.parent / record_name))
        # emg = record.p_signal
        # 
        # # Select channel group
        # start, end = self.CHANNEL_GROUPS[self.channel_group]
        # emg = emg[:, start:end]
        # 
        # # Extract metadata from filename
        # metadata = self._extract_metadata(path)
        # label = metadata.get("gesture", 0)
        # 
        # return EMGData(emg=emg, label=label, metadata=metadata)
    
    def get_label(self, path: Path) -> int:
        """Extract class label from filename.
        
        Filename pattern: session{i}_participant{j}_gesture{k}_trial{l}
        
        Args:
            path: Path to the data file.
            
        Returns:
            Integer class label (gesture number).
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError("MyoLoader.get_label() is not yet implemented.")
        
        # TODO: Implementation outline:
        # import re
        # match = re.search(r"gesture(\d+)", path.stem)
        # if match:
        #     return int(match.group(1))
        # return 0
    
    def _extract_metadata(self, path: Path) -> dict:
        """Extract metadata from filename.
        
        Filename pattern: session{i}_participant{j}_gesture{k}_trial{l}
        
        Args:
            path: Path to the data file.
            
        Returns:
            Dictionary with extracted metadata.
        """
        import re
        
        name = path.stem
        metadata = {
            "filename": path.name,
            "filepath": str(path),
        }
        
        # Parse filename components
        # Pattern: session{i}_participant{j}_gesture{k}_trial{l}
        session_match = re.search(r"session(\d+)", name)
        participant_match = re.search(r"participant(\d+)", name)
        gesture_match = re.search(r"gesture(\d+)", name)
        trial_match = re.search(r"trial(\d+)", name)
        
        if session_match:
            metadata["session"] = int(session_match.group(1))
        if participant_match:
            subject_num = int(participant_match.group(1))
            metadata["subject"] = f"s{subject_num:02d}"
        if gesture_match:
            metadata["gesture"] = int(gesture_match.group(1))
            metadata["position"] = f"G{metadata['gesture']}"  # Use gesture as position
        if trial_match:
            metadata["trial"] = int(trial_match.group(1))
            metadata["repetition"] = metadata["trial"]
        
        return metadata
    
    def discover_files(self) -> List[Path]:
        """Find all WFDB .dat files matching the pattern.
        
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
            import re
            filtered = []
            for f in files:
                match = re.search(r"session(\d+)", f.stem)
                if match and int(match.group(1)) in self.sessions:
                    filtered.append(f)
            files = filtered
        
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
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError("MyoLoader.load_by_subject() is not yet implemented.")
    
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
            
        Raises:
            NotImplementedError: This is a stub.
        """
        raise NotImplementedError("MyoLoader.load_by_session() is not yet implemented.")
