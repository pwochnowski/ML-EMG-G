"""CLI for feature extraction from EMG data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import numpy as np

from ..config import load_config, PipelineConfig, WindowConfig, FeatureConfig, DatasetConfig
from ..data.loaders import TextFileLoader, MatFileLoader
from ..data.windowing import sliding_window
from ..features import extract_spectral_features, extract_time_features, extract_combined_features


def get_extractor(config: FeatureConfig, sampling_rate: float):
    """Get the feature extraction function based on config."""
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process EMG files for a single subject.
    
    Returns:
        Tuple of (X, y, files, positions) arrays.
    """
    # Create loader
    loader = TextFileLoader(
        data_dir=subject_dir,
        n_channels=dataset_config.n_channels,
        sampling_rate=dataset_config.sampling_rate,
        file_pattern=dataset_config.file_pattern,
        label_mapping=dataset_config.label_mapping or None,
    )
    
    # Get feature extractor
    extractor = get_extractor(feature_config, dataset_config.sampling_rate)
    
    # Load data by position
    data_by_pos = loader.load_by_position(positions=positions, limit_per_position=limit)
    
    X_list = []
    y_list = []
    src_files = []
    src_positions = []
    
    for pos, recordings in data_by_pos.items():
        for emg_data in recordings:
            # Apply sliding window
            windows = sliding_window(
                emg_data.emg,
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
        
    Returns:
        Tuple of (X, y, groups, positions) arrays.
    """
    # Create MAT loader
    emg_columns = None
    if dataset_config.emg_columns:
        emg_columns = tuple(dataset_config.emg_columns)
    
    loader = MatFileLoader(
        data_dir=data_dir,
        n_channels=dataset_config.n_channels,
        sampling_rate=dataset_config.sampling_rate,
        file_pattern=dataset_config.file_pattern,
        label_column=dataset_config.label_column or "restimulus",
        emg_columns=emg_columns,
    )
    
    # Get feature extractor
    extractor = get_extractor(feature_config, dataset_config.sampling_rate)
    
    # Load data grouped by subject
    data_by_subject = loader.load_by_subject(subjects=subjects)
    
    all_X = []
    all_y = []
    all_groups = []
    all_positions = []
    
    for subject, recordings in sorted(data_by_subject.items()):
        subject_X = []
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
            
            # Apply sliding window to both EMG and labels
            emg_windows = sliding_window(
                emg_data.emg,
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
                
                # Extract features
                feats = extractor(emg_windows[i])
                subject_X.append(feats)
                subject_y.append(label)
                subject_pos.append(position)
        
        if len(subject_X) == 0:
            print(f"Warning: No features extracted for subject {subject}")
            continue
        
        # Stack subject data
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
            out_path = output_dir / f"{subject}_features.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=y,
                groups=groups,
                positions=positions_arr,
            )
            print(f"Saved: {out_path} (X.shape={X.shape}, classes={len(np.unique(y))})")
    
    if len(all_X) == 0:
        raise RuntimeError("No features extracted — check data paths and config")
    
    # Combine all subjects
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    groups_combined = np.concatenate(all_groups)
    positions_combined = np.concatenate(all_positions)
    
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
        help="Feature set to extract"
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
    
    parsed = parser.parse_args(args)
    
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
    
    # Route based on loader type
    if dataset_config.loader_type == "mat":
        # Parse subjects and exercises
        subjects = parsed.subjects.split(",") if parsed.subjects else None
        exercises = [int(e) for e in parsed.exercises.split(",")] if parsed.exercises else None
        
        # Process continuous data (NinaPro-style)
        X, y, groups, positions = process_continuous_data(
            data_dir=dataset_config.data_dir,
            dataset_config=dataset_config,
            window_config=window_config,
            feature_config=feature_config,
            output_dir=output_dir,
            subjects=subjects,
            exercises=exercises,
            include_rest=parsed.include_rest,
        )
        
        print(f"\nTotal: X.shape={X.shape}, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects")
    else:
        # Original text file processing
        subject_dir = Path(parsed.subject) if parsed.subject else dataset_config.data_dir
        output_path = parsed.out
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if output_path:
                output_path = output_dir / output_path.name
            else:
                output_path = output_dir / f"{subject_dir.name}_features.npz"
        elif output_path is None:
            output_path = Path(f"{subject_dir.name}_features.npz")
        
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
        )


if __name__ == "__main__":
    main()
