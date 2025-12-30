"""CLI for feature extraction from EMG data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from ..config import load_config, PipelineConfig, WindowConfig, FeatureConfig, DatasetConfig
from ..config.schema import load_feature_set, list_feature_sets, FeatureSetConfig, load_preprocessing_config, PreprocessingConfig
from ..data.loaders import get_loader, DB1Loader, RamiLoader
from ..data.windowing import sliding_window
from ..features import extract_spectral_features, extract_time_features, extract_combined_features
from ..features.combined import extract_features_from_config
from ..features.gpu import CUPY_AVAILABLE, extract_features_batch_gpu
from ..preprocessing import SignalPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_extractor(config: FeatureConfig, sampling_rate: float, feature_set_config: Optional[FeatureSetConfig] = None):
    """Get the feature extraction function based on config.
    
    Args:
        config: FeatureConfig with extractor_type.
        sampling_rate: Sampling rate in Hz.
        feature_set_config: Optional FeatureSetConfig for named feature sets (experimental, full, etc.)
    """
    # If we have a named feature set config, use it for full feature extraction
    if feature_set_config is not None:
        def extractor(window):
            return extract_features_from_config(window, feature_set_config, sampling_rate=sampling_rate)
        return extractor
    
    if config.extractor_type == "spectral":
        def extractor(window):
            return extract_spectral_features(
                window,
                sampling_rate=sampling_rate,
                include_moments=config.spectral_moments,
                include_flux=config.spectral_flux,
                include_sparsity=config.spectral_sparsity,
                include_irregularity=config.irregularity_factor,
                include_correlation=config.spectrum_correlation,
            )
        return extractor
    elif config.extractor_type == "time":
        def extractor(window):
            return extract_time_features(
                window,
                threshold=config.time_threshold,
                include_mav=config.time_mav,
                include_wl=config.time_wl,
                include_zc=config.time_zc,
                include_ssc=config.time_ssc,
            )
        return extractor
    else:  # combined
        def extractor(window):
            return extract_combined_features(
                window,
                sampling_rate=sampling_rate,
                threshold=config.time_threshold,
            )
        return extractor


def process_subject(
    subject_dir: Path,
    dataset_config: DatasetConfig,
    window_config: WindowConfig,
    feature_config: FeatureConfig,
    output_path: Optional[Path] = None,
    positions: Optional[List[str]] = None,
    limit: int = 0,
    subject_mapping: Optional[dict] = None,
    feature_set_config: Optional[FeatureSetConfig] = None,
    preprocessor: Optional[SignalPreprocessor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process EMG files for a single subject.
    
    Args:
        subject_dir: Path to subject data directory.
        dataset_config: Dataset configuration.
        window_config: Windowing configuration.
        feature_config: Basic feature configuration.
        output_path: Optional path to save output.
        positions: Optional list of positions to process.
        limit: Limit files per position (0 = no limit).
        subject_mapping: Optional mapping of folder names to subject IDs.
        feature_set_config: Optional named feature set config (e.g., 'experimental').
        preprocessor: Optional signal preprocessor for filtering.
    
    Returns:
        Tuple of (X, y, files, positions) arrays.
    """
    # Create loader using factory function
    LoaderClass = get_loader(dataset_config.loader_type or "rami")
    
    # Build loader kwargs based on loader type
    loader_kwargs = {
        "data_dir": subject_dir,
        "n_channels": dataset_config.n_channels,
        "sampling_rate": dataset_config.sampling_rate,
        "file_pattern": dataset_config.file_pattern,
    }
    
    # Add loader-specific parameters
    if dataset_config.loader_type in ("rami"):
        loader_kwargs["label_mapping"] = dataset_config.label_mapping or None
        loader_kwargs["subject_mapping"] = subject_mapping
    
    loader = LoaderClass(**loader_kwargs)
    
    # Get feature extractor (use named feature set if provided)
    extractor = get_extractor(feature_config, dataset_config.sampling_rate, feature_set_config)
    
    # Load data by position
    data_by_pos = loader.load_by_position(positions=positions, limit_per_position=limit)
    
    X_list = []
    y_list = []
    src_files = []
    src_positions = []
    
    for pos, recordings in data_by_pos.items():
        for emg_data in recordings:
            # Apply preprocessing if configured
            emg_signal = emg_data.emg
            if preprocessor is not None:
                emg_signal = preprocessor.process(emg_signal, fs=dataset_config.sampling_rate)
            
            # Apply sliding window
            windows = sliding_window(
                emg_signal,
                window_size=window_config.size,
                window_increment=window_config.increment,
            )
            
            if windows.shape[0] == 0:
                continue
            
            # Extract features from each window
            for w in windows:
                feats = extractor(w)
                X_list.append(feats)
                y_list.append(emg_data.label)
                src_files.append(emg_data.metadata.get("filename", "unknown"))
                src_positions.append(pos)
    
    if len(X_list) == 0:
        raise RuntimeError("No features extracted — check paths/positions/winsize")
    
    X = np.vstack([f.reshape(1, -1) if f.ndim == 1 else f for f in X_list])
    y = np.array(y_list, dtype=np.int64)
    files = np.array(src_files)
    positions_arr = np.array(src_positions)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            files=files,
            positions=positions_arr,
        )
        print(f"Saved features to: {output_path} (X.shape={X.shape})")
    
    return X, y, files, positions_arr


def process_continuous_data(
    data_dir: Path,
    dataset_config: DatasetConfig,
    window_config: WindowConfig,
    feature_config: FeatureConfig,
    output_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    exercises: Optional[List[int]] = None,
    include_rest: bool = True,
    feature_set_config: Optional[FeatureSetConfig] = None,
    preprocessor: Optional[SignalPreprocessor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process continuous EMG data with per-sample labels (e.g., NinaPro).
    
    This handles datasets where each file contains continuous EMG with
    per-sample stimulus labels, rather than one label per file.
    
    Args:
        data_dir: Directory containing data files.
        dataset_config: Dataset configuration.
        window_config: Windowing configuration.
        feature_config: Feature extraction configuration.
        output_dir: Directory to save output files.
        subjects: List of subject IDs to process (None = all).
        exercises: List of exercise numbers to include (None = all).
        include_rest: Whether to include rest class (label 0).
        feature_set_config: Optional named feature set config (e.g., 'experimental').
        preprocessor: Optional signal preprocessor for filtering.
        
    Returns:
        Tuple of (X, y, groups, positions) arrays.
    """
    # Create loader using factory function
    LoaderClass = get_loader(dataset_config.loader_type or "db1")
    
    # Build loader kwargs based on loader type
    loader_kwargs = {
        "data_dir": data_dir,
        "n_channels": dataset_config.n_channels,
        "sampling_rate": dataset_config.sampling_rate,
        "file_pattern": dataset_config.file_pattern,
    }
    
    # Add loader-specific parameters
    if dataset_config.loader_type in ("db1", "mat"):
        if dataset_config.emg_columns:
            loader_kwargs["emg_columns"] = tuple(dataset_config.emg_columns)
        loader_kwargs["label_column"] = dataset_config.label_column or "restimulus"
    
    loader = LoaderClass(**loader_kwargs)
    
    # Get feature extractor (use named feature set if provided)
    extractor = get_extractor(feature_config, dataset_config.sampling_rate, feature_set_config)
    
    # Load data grouped by subject
    data_by_subject = loader.load_by_subject(subjects=subjects)
    
    all_X = []
    all_y = []
    all_groups = []
    all_positions = []
    
    # Check if GPU is available for batch processing
    use_gpu = False and CUPY_AVAILABLE and feature_set_config is not None
    
    for subject, recordings in sorted(data_by_subject.items()):
        # Check if subject already exists
        if output_dir:
            out_path = output_dir / f"{subject}.npz"
            if out_path.exists():
                print(f"Skipping {subject}: {out_path} already exists")
                # Load existing data to include in combined output
                data = np.load(out_path)
                all_X.append(data['X'])
                all_y.append(data['y'])
                all_groups.append(data['groups'])
                all_positions.append(data['positions'])
                continue
        
        # Collect all windows for this subject first (for batch GPU processing)
        subject_windows = []
        subject_y = []
        subject_pos = []
        
        for emg_data in recordings:
            # Filter by exercise if specified
            exercise = emg_data.metadata.get("exercise", 0)
            if exercises is not None and exercise not in exercises:
                continue
            
            # Get per-sample labels
            sample_labels = emg_data.metadata.get("sample_labels")
            if sample_labels is None:
                print(f"Warning: No sample labels for {emg_data.metadata.get('filename')}")
                continue
            
            position = emg_data.metadata.get("position", f"E{exercise}")
            
            # Apply preprocessing if configured
            emg_signal = emg_data.emg
            if preprocessor is not None:
                emg_signal = preprocessor.process(emg_signal, fs=dataset_config.sampling_rate)
            
            # Apply sliding window to both EMG and labels
            emg_windows = sliding_window(
                emg_signal,
                window_size=window_config.size,
                window_increment=window_config.increment,
            )
            
            if emg_windows.shape[0] == 0:
                continue
            
            # Get window labels (use mode/majority of samples in each window)
            n_windows = emg_windows.shape[0]
            for i in range(n_windows):
                start_idx = i * window_config.increment
                end_idx = start_idx + window_config.size
                window_labels = sample_labels[start_idx:end_idx]
                
                # Use majority vote for window label
                label = int(np.bincount(window_labels.astype(int)).argmax())
                
                # Skip rest windows if not included
                if not include_rest and label == 0:
                    continue
                
                # Collect window and label
                subject_windows.append(emg_windows[i])
                subject_y.append(label)
                subject_pos.append(position)
        
        if len(subject_windows) == 0:
            print(f"Warning: No features extracted for subject {subject}")
            continue
        
        # Extract features - use GPU batch processing if available
        if use_gpu:
            windows_batch = np.array(subject_windows, dtype=np.float32)
            X = extract_features_batch_gpu(
                windows_batch,
                sampling_rate=dataset_config.sampling_rate,
                config=feature_set_config,
            )
        else:
            subject_X = [extractor(w) for w in subject_windows]
            X = np.vstack([f.reshape(1, -1) if f.ndim == 1 else f for f in subject_X])
        
        y = np.array(subject_y, dtype=np.int64)
        positions_arr = np.array(subject_pos)
        groups = np.array([subject] * len(y))
        
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
        all_positions.append(positions_arr)
        
        # Optionally save per-subject file
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{subject}.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=y,
                groups=groups,
                positions=positions_arr,
            )
            print(f"Saved: {out_path} (X.shape={X.shape}, classes={len(np.unique(y))})")
    
    if len(all_X) == 0:
        # Check if we skipped all subjects because they already exist
        if output_dir and any((output_dir / f"{s}.npz").exists() for s in data_by_subject.keys()):
            # Load existing data
            for subject in sorted(data_by_subject.keys()):
                out_path = output_dir / f"{subject}.npz"
                if out_path.exists():
                    data = np.load(out_path)
                    all_X.append(data['X'])
                    all_y.append(data['y'])
                    all_groups.append(data['groups'])
                    all_positions.append(data['positions'])
        else:
            raise RuntimeError("No features extracted — check data paths and config")
    
    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    groups_combined = np.concatenate(all_groups)
    positions_combined = np.concatenate(all_positions)
    
    return X_combined, y_combined, groups_combined, positions_combined


def process_discrete_data(
    data_dir: Path,
    dataset_config: DatasetConfig,
    window_config: WindowConfig,
    feature_config: FeatureConfig,
    output_dir: Optional[Path] = None,
    feature_set_config: Optional[FeatureSetConfig] = None,
    preprocessor: Optional[SignalPreprocessor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process discrete EMG data with per-file labels (e.g., Myo, Rami).
    
    This handles datasets where each file contains one trial with a single
    label for the entire file.
    
    Args:
        data_dir: Directory containing data files.
        dataset_config: Dataset configuration.
        window_config: Windowing configuration.
        feature_config: Feature extraction configuration.
        output_dir: Directory to save output files.
        feature_set_config: Optional named feature set config (e.g., 'experimental').
        preprocessor: Optional signal preprocessor for filtering.
        
    Returns:
        Tuple of (X, y, groups, positions) arrays.
    """
    logger.info(f"Starting discrete data processing from: {data_dir}")
    logger.info(f"  data_dir exists: {data_dir.exists()}")
    logger.info(f"  data_dir is_symlink: {data_dir.is_symlink()}")
    if data_dir.is_symlink():
        logger.info(f"  symlink target: {data_dir.resolve()}")
    
    # Create loader using factory function
    LoaderClass = get_loader(dataset_config.loader_type)
    logger.info(f"Using loader: {LoaderClass.__name__}")
    
    # Build loader kwargs based on loader type
    loader_kwargs = {
        "data_dir": data_dir,
        "n_channels": dataset_config.n_channels,
        "sampling_rate": dataset_config.sampling_rate,
        "file_pattern": dataset_config.file_pattern,
    }
    logger.info(f"Loader kwargs: n_channels={dataset_config.n_channels}, "
                f"sampling_rate={dataset_config.sampling_rate}, "
                f"file_pattern={dataset_config.file_pattern}")
    
    # Add loader-specific parameters
    if dataset_config.loader_type == "rami":
        loader_kwargs["label_mapping"] = dataset_config.label_mapping or None
        # Get subject mapping from config
        subjects_config = getattr(dataset_config, 'subjects', None)
        if subjects_config and isinstance(subjects_config, dict):
            loader_kwargs["subject_mapping"] = subjects_config
    elif dataset_config.loader_type == "myo":
        loader_kwargs["channel_group"] = getattr(dataset_config, 'channel_group', 'forearm')
        loader_kwargs["sessions"] = getattr(dataset_config, 'sessions', None)
        logger.info(f"Myo loader: channel_group={loader_kwargs.get('channel_group')}, "
                    f"sessions={loader_kwargs.get('sessions')}")
    
    loader = LoaderClass(**loader_kwargs)
    
    # Get feature extractor (use named feature set if provided)
    extractor = get_extractor(feature_config, dataset_config.sampling_rate, feature_set_config)
    
    # Load data grouped by subject (all subjects)
    logger.info("Loading data by subject...")
    load_start = time.time()
    data_by_subject = loader.load_by_subject(subjects=None)
    load_time = time.time() - load_start
    logger.info(f"Loaded {len(data_by_subject)} subjects in {load_time:.2f}s")
    
    all_X = []
    all_y = []
    all_groups = []
    all_positions = []
    
    # Check if GPU is available for batch processing
    use_gpu = False and CUPY_AVAILABLE and feature_set_config is not None
    if use_gpu:
        logger.info(f"Processing {len(data_by_subject)} subjects (GPU accelerated)")
    else:
        logger.info(f"Processing {len(data_by_subject)} subjects (CPU)")
    
    total_subjects = len(data_by_subject)
    for subj_idx, (subject, recordings) in enumerate(sorted(data_by_subject.items())):
        # Check if subject already exists
        if output_dir:
            out_path = output_dir / f"{subject}.npz"
            if out_path.exists():
                logger.info(f"[{subj_idx+1}/{total_subjects}] Skipping {subject}: {out_path} already exists")
                # Load existing data to include in combined output
                data = np.load(out_path)
                all_X.append(data['X'])
                all_y.append(data['y'])
                all_groups.append(data['groups'])
                all_positions.append(data['positions'])
                continue
        
        subj_start = time.time()
        logger.info(f"[{subj_idx+1}/{total_subjects}] Processing subject {subject} ({len(recordings)} recordings)")
        
        # Collect all windows for this subject first (for batch GPU processing)
        subject_windows = []
        subject_y = []
        subject_pos = []
        
        for rec_idx, emg_data in enumerate(recordings):
            position = emg_data.metadata.get("position", "unknown")
            
            # Apply preprocessing if configured
            emg_signal = emg_data.emg
            if preprocessor is not None:
                emg_signal = preprocessor.process(emg_signal, fs=dataset_config.sampling_rate)
            
            # Apply sliding window
            emg_windows = sliding_window(
                emg_signal,
                window_size=window_config.size,
                window_increment=window_config.increment,
            )
            
            if emg_windows.shape[0] == 0:
                logger.debug(f"  Recording {rec_idx}: 0 windows (signal too short)")
                continue
            
            # Collect windows and labels
            for w in emg_windows:
                subject_windows.append(w)
                subject_y.append(emg_data.label)
                subject_pos.append(position)
        
        if len(subject_windows) == 0:
            logger.warning(f"No features extracted for subject {subject}")
            continue
        
        logger.info(f"  Collected {len(subject_windows)} windows, extracting features...")
        feat_start = time.time()
        
        # Extract features - use GPU batch processing if available
        if use_gpu:
            # Stack windows for batch GPU processing
            windows_batch = np.array(subject_windows, dtype=np.float32)
            X = extract_features_batch_gpu(
                windows_batch,
                sampling_rate=dataset_config.sampling_rate,
                config=feature_set_config,
            )
        else:
            # CPU extraction window-by-window
            subject_X = [extractor(w) for w in subject_windows]
            X = np.vstack([f.reshape(1, -1) if f.ndim == 1 else f for f in subject_X])
        
        feat_time = time.time() - feat_start
        subj_time = time.time() - subj_start
        logger.info(f"  Features extracted: X.shape={X.shape} in {feat_time:.2f}s (total: {subj_time:.2f}s)")
        
        y = np.array(subject_y, dtype=np.int64)
        positions_arr = np.array(subject_pos)
        groups = np.array([subject] * len(y))
        
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
        all_positions.append(positions_arr)
        
        # Optionally save per-subject file
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{subject}.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=y,
                groups=groups,
                positions=positions_arr,
            )
            logger.info(f"  Saved: {out_path}")
    
    if len(all_X) == 0:
        raise RuntimeError("No features extracted — check data paths and config")
    
    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    groups_combined = np.concatenate(all_groups)
    positions_combined = np.concatenate(all_positions)
    
    logger.info(f"Total: X.shape={X_combined.shape}, {len(np.unique(y_combined))} classes, "
                f"{len(np.unique(groups_combined))} subjects")
    
    return X_combined, y_combined, groups_combined, positions_combined


def main(args: Optional[List[str]] = None):
    """Main entry point for feature extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Extract features from EMG data files"
    )
    parser.add_argument(
        "subject",
        nargs="?",
        help="Subject directory (e.g., data/S1_Male) for text files, or data root for MAT files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output .npz file path"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--winsize",
        type=int,
        default=400,
        help="Window size in samples"
    )
    parser.add_argument(
        "--wininc",
        type=int,
        default=100,
        help="Window increment in samples"
    )
    parser.add_argument(
        "--positions",
        help="Comma-separated positions to include (e.g., Pos1,Pos2)"
    )
    parser.add_argument(
        "--exercises",
        help="Comma-separated exercises to include for MAT files (e.g., 1,2,3)"
    )
    parser.add_argument(
        "--subjects",
        help="Comma-separated subjects to include (e.g., s01,s02)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit files per position (0 = no limit)"
    )
    parser.add_argument(
        "--feat",
        choices=["spectral", "time", "combined"],
        default="combined",
        help="Feature set to extract (basic mode)"
    )
    parser.add_argument(
        "--feature-set",
        help="Named feature set from features.yaml (e.g., 'default', 'experimental', 'full'). "
             "Overrides --feat when specified. Use --list-features to see available options."
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List available feature sets from features.yaml and exit"
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=7,
        help="Number of EMG channels"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1000.0,
        help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--include-rest",
        action="store_true",
        help="Include rest class (label 0) for continuous data"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply preprocessing (bandpass + notch filter) before feature extraction. "
             "Uses settings from features.yaml or defaults (20-500Hz bandpass, 50Hz notch)."
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=None,
        help="Path to a custom features.yaml file for preprocessing settings. "
             "Allows running multiple configs in parallel without conflicts."
    )
    parser.add_argument(
        "--subset-name",
        default="all",
        help="Subset name for output directory (e.g., 'all', 'e1', 'pos1'). Output: features/{extractor_type}/{subset_name}/{subject}.npz"
    )
    
    parsed = parser.parse_args(args)
    
    # Handle --list-features
    if parsed.list_features:
        print("Available feature sets (defined in src/emg_classification/config/features.yaml):")
        for name in list_feature_sets():
            print(f"  - {name}")
        print("\nUse --feature-set <name> to extract features with a specific configuration.")
        return
    
    # Load named feature set if specified
    feature_set_config = None
    if parsed.feature_set:
        try:
            feature_set_config = load_feature_set(parsed.feature_set)
            print(f"Using feature set: {parsed.feature_set}")
        except ValueError as e:
            parser.error(str(e))
    
    # Set up preprocessor if requested
    preprocessor = None
    if parsed.preprocess:
        preprocess_config = load_preprocessing_config(parsed.preprocess_config)
        preprocessor = SignalPreprocessor(preprocess_config)
        print(f"Preprocessing enabled: {preprocessor}")
    
    # Build configs from CLI args or load from file
    if parsed.config:
        config = load_config(parsed.config)
        dataset_config = config.dataset
        window_config = config.window
        feature_config = config.features
        results_dir = config.results_dir
    else:
        if not parsed.subject:
            parser.error("subject is required when not using --config")
        dataset_config = DatasetConfig(
            name="cli",
            n_channels=parsed.n_channels,
            sampling_rate=parsed.sampling_rate,
            data_dir=Path(parsed.subject),
            file_pattern="Pos*_*.txt",
        )
        window_config = WindowConfig(size=parsed.winsize, increment=parsed.wininc)
        feature_config = FeatureConfig(extractor_type=parsed.feat)
        results_dir = Path("results")
    
    # Determine output directory
    output_dir = parsed.out_dir or results_dir
    
    # Determine the feature type directory name
    if feature_set_config:
        feature_type_dir = parsed.feature_set  # Use feature set name as directory
    else:
        feature_type_dir = feature_config.extractor_type
    
    # Log configuration summary
    logger.info("=" * 60)
    logger.info("EMG Feature Extraction")
    logger.info("=" * 60)
    logger.info(f"Loader type: {dataset_config.loader_type}")
    logger.info(f"Data dir: {dataset_config.data_dir}")
    logger.info(f"Window: size={window_config.size}, increment={window_config.increment}")
    logger.info(f"Feature set: {parsed.feature_set or feature_config.extractor_type}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"GPU available: {CUPY_AVAILABLE}")
    if preprocessor:
        logger.info(f"Preprocessing: enabled")
    logger.info("=" * 60)
    
    # Route based on loader type
    # db1 has continuous per-sample labels, others have per-file labels
    if dataset_config.loader_type == "db1":
        # Parse subjects and exercises
        subjects = parsed.subjects.split(",") if parsed.subjects else None
        exercises = [int(e) for e in parsed.exercises.split(",")] if parsed.exercises else None
        
        # Build nested output directory: features/{feature_type}/{subset_name}/
        subset_output_dir = output_dir / feature_type_dir / parsed.subset_name
        
        # Process continuous data (NinaPro-style)
        X, y, groups, positions = process_continuous_data(
            data_dir=dataset_config.data_dir,
            dataset_config=dataset_config,
            window_config=window_config,
            feature_config=feature_config,
            output_dir=subset_output_dir,
            subjects=subjects,
            exercises=exercises,
            include_rest=parsed.include_rest,
            feature_set_config=feature_set_config,
            preprocessor=preprocessor,
        )
        
        print(f"\nTotal: X.shape={X.shape}, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects")
    elif dataset_config.loader_type in ("myo", "rami"):
        # myo/rami: use load_by_subject to extract all subjects at once
        # --subjects argument is ignored, extracts all subjects
        if parsed.subjects:
            print(f"Note: --subjects argument ignored for {dataset_config.loader_type}, extracting all subjects")
        
        # Build nested output directory: features/{feature_type}/{subset_name}/
        subset_output_dir = output_dir / feature_type_dir / parsed.subset_name
        
        # Process discrete data (per-file labels)
        X, y, groups, positions = process_discrete_data(
            data_dir=dataset_config.data_dir,
            dataset_config=dataset_config,
            window_config=window_config,
            feature_config=feature_config,
            output_dir=subset_output_dir,
            feature_set_config=feature_set_config,
            preprocessor=preprocessor,
        )
        
        print(f"\nTotal: X.shape={X.shape}, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects")
    else:
        # Original text file processing
        subject_dir = Path(parsed.subject) if parsed.subject else dataset_config.data_dir
        output_path = parsed.out
        
        # Get subject mapping and standardized subject ID from config
        subjects_config = getattr(dataset_config, 'subjects', None)
        subject_mapping = None
        folder_name = subject_dir.name
        
        if subjects_config and isinstance(subjects_config, dict):
            # Use mapping: folder_name -> subject_id
            subject_mapping = subjects_config
            subject_id = subjects_config.get(folder_name, folder_name)
        else:
            subject_id = folder_name
        
        if output_dir:
            # Build nested output directory: features/{feature_type}/{subset_name}/
            subset_output_dir = output_dir / feature_type_dir / parsed.subset_name
            subset_output_dir.mkdir(parents=True, exist_ok=True)
            if output_path:
                output_path = subset_output_dir / output_path.name
            else:
                output_path = subset_output_dir / f"{subject_id}.npz"
        elif output_path is None:
            output_path = Path(f"{subject_id}.npz")
        
        # Check if output already exists
        if output_path.exists():
            logger.info(f"Skipping: {output_path} already exists")
            return
        
        # Parse positions
        positions = parsed.positions.split(",") if parsed.positions else None
        
        # Process
        process_subject(
            subject_dir=subject_dir,
            dataset_config=dataset_config,
            window_config=window_config,
            feature_config=feature_config,
            output_path=output_path,
            positions=positions,
            limit=parsed.limit,
            subject_mapping=subject_mapping,
            feature_set_config=feature_set_config,
            preprocessor=preprocessor,
        )


if __name__ == "__main__":
    main()
