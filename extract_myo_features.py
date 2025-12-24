"""Extract features from GrabMyo dataset.

Loads EMG data via MyoLoader, applies windowing, extracts time-domain features,
and saves per-subject .npz files with session metadata for cross-session evaluation.

Usage:
    # Extract features from all sessions, forearm channels only
    uv run extract_myo_features.py --channel-group forearm
    
    # Extract from specific sessions
    uv run extract_myo_features.py --sessions 1 2 --channel-group forearm
    
    # Extract from specific subjects
    uv run extract_myo_features.py --subjects s01 s02 s03
    
    # Custom window size and overlap
    uv run extract_myo_features.py --window-size 256 --overlap 128
"""

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm

from src.emg_classification.data.loaders.myo_loader import MyoLoader
from time_features import extract_time_features


def window_signal(emg: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
    """Split EMG signal into overlapping windows.
    
    Args:
        emg: Signal of shape (n_samples, n_channels)
        window_size: Number of samples per window
        overlap: Number of overlapping samples between windows
        
    Returns:
        Array of shape (n_windows, window_size, n_channels)
    """
    step = window_size - overlap
    n_samples = emg.shape[0]
    n_windows = (n_samples - window_size) // step + 1
    
    if n_windows <= 0:
        # Signal too short, return single padded window
        padded = np.zeros((window_size, emg.shape[1]), dtype=emg.dtype)
        padded[:n_samples] = emg
        return padded[np.newaxis, ...]
    
    windows = []
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        windows.append(emg[start:end])
    
    return np.array(windows)


def extract_features_from_window(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Extract features from a single window.
    
    Args:
        window: EMG window of shape (window_size, n_channels)
        threshold: Threshold for ZC and SSC features
        
    Returns:
        1D feature vector
    """
    return extract_time_features(window, thres=threshold)


def process_subject(
    loader: MyoLoader,
    subject_id: str,
    window_size: int,
    overlap: int,
    threshold: float = 0.01,
) -> dict:
    """Process all files for a single subject.
    
    Args:
        loader: MyoLoader instance
        subject_id: Subject ID (e.g., 's01')
        window_size: Window size in samples
        overlap: Overlap in samples
        threshold: Feature extraction threshold
        
    Returns:
        Dictionary with X, y, sessions, gestures, trials, positions arrays
    """
    # Load all data for this subject
    subject_data = loader.load_by_subject(subjects=[subject_id])
    
    if subject_id not in subject_data:
        return None
    
    all_features = []
    all_labels = []
    all_sessions = []
    all_gestures = []
    all_trials = []
    all_positions = []
    all_files = []
    
    for emg_data in subject_data[subject_id]:
        # Window the signal
        windows = window_signal(emg_data.emg, window_size, overlap)
        
        # Extract features from each window
        for window in windows:
            features = extract_features_from_window(window, threshold)
            all_features.append(features)
            all_labels.append(emg_data.label)
            all_sessions.append(emg_data.metadata.get('session', 0))
            all_gestures.append(emg_data.metadata.get('gesture', 0))
            all_trials.append(emg_data.metadata.get('trial', 0))
            all_positions.append(emg_data.metadata.get('position', 'unknown'))
            all_files.append(emg_data.metadata.get('filename', ''))
    
    if not all_features:
        return None
    
    return {
        'X': np.array(all_features, dtype=np.float32),
        'y': np.array(all_labels, dtype=np.int32),
        'sessions': np.array(all_sessions, dtype=np.int32),
        'gestures': np.array(all_gestures, dtype=np.int32),
        'trials': np.array(all_trials, dtype=np.int32),
        'positions': np.array(all_positions, dtype=object),
        'files': np.array(all_files, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description='Extract features from GrabMyo dataset')
    parser.add_argument('--data-dir', default='datasets/myo/data',
                        help='Path to GrabMyo data directory')
    parser.add_argument('--output-dir', default='datasets/myo/features',
                        help='Output directory for feature files')
    parser.add_argument('--channel-group', default='forearm',
                        choices=['forearm', 'wrist', 'all'],
                        help='Which electrode group to use')
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Sessions to include (e.g., 1 2 3). Default: all')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subjects to process (e.g., s01 s02). Default: all')
    parser.add_argument('--window-size', type=int, default=256,
                        help='Window size in samples (default: 256 = 125ms at 2048Hz)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Window overlap in samples (default: 128 = 50%% overlap)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Threshold for ZC/SSC features')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.channel_group
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loader
    print(f"Loading GrabMyo data from: {args.data_dir}")
    print(f"Channel group: {args.channel_group}")
    print(f"Sessions: {args.sessions or 'all'}")
    print(f"Window: {args.window_size} samples, {args.overlap} overlap")
    print(f"Sampling rate: 2048 Hz -> window = {args.window_size/2048*1000:.1f}ms")
    
    loader = MyoLoader(
        data_dir=args.data_dir,
        channel_group=args.channel_group,
        sessions=args.sessions,
    )
    
    # Get subjects to process
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = loader.get_subject_ids()
    
    print(f"Processing {len(subjects)} subjects...")
    
    # Process each subject
    for subject_id in tqdm(subjects, desc="Subjects"):
        result = process_subject(
            loader=loader,
            subject_id=subject_id,
            window_size=args.window_size,
            overlap=args.overlap,
            threshold=args.threshold,
        )
        
        if result is None:
            print(f"Warning: No data found for {subject_id}")
            continue
        
        # Save to .npz file
        output_path = output_dir / f"{subject_id}.npz"
        np.savez_compressed(
            output_path,
            X=result['X'],
            y=result['y'],
            sessions=result['sessions'],
            gestures=result['gestures'],
            trials=result['trials'],
            positions=result['positions'],
            files=result['files'],
            # Metadata
            subject=subject_id,
            channel_group=args.channel_group,
            window_size=args.window_size,
            overlap=args.overlap,
            sampling_rate=2048.0,
        )
        
        n_samples = result['X'].shape[0]
        n_features = result['X'].shape[1]
        unique_sessions = np.unique(result['sessions'])
        tqdm.write(f"  {subject_id}: {n_samples} windows, {n_features} features, sessions {list(unique_sessions)}")
    
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Use with loso_train.py:")
    print(f"  uv run loso_train.py --dataset myo --feature-subdir {args.channel_group}")


if __name__ == '__main__':
    main()
