"""Feature extraction script.

Reads EMG `.txt` files in a subject folder (e.g. `S1_Male/`), windows them,
extracts either spectral (`khushaba`) or time-domain features, and saves a
compressed `.npz` with `X`, `y`, `files`, and `positions`.
"""

from pathlib import Path
import argparse
import numpy as np
import yaml
from rami_features import extract_khushaba_features
from time_features import extract_time_features


def load_dataset_config(dataset_name: str) -> dict:
    """Load config.yaml for a dataset and return resolved output paths."""
    config_path = Path(f"datasets/{dataset_name}/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Resolve output directories relative to dataset root
    dataset_root = Path(f"datasets/{dataset_name}")
    output = config.get("output", {})
    config["_resolved"] = {
        "features_dir": dataset_root / output.get("features_dir", "features"),
        "training_dir": dataset_root / output.get("training_dir", "training"),
        "tuning_dir": dataset_root / output.get("tuning_dir", "tuning"),
        "models_dir": dataset_root / output.get("models_dir", "models"),
        "analysis_dir": dataset_root / output.get("analysis_dir", "analysis"),
        "data_dir": dataset_root / "data",
    }
    return config


def sliding_window(x, winsize, wininc):
    """Return array of overlapping windows [num_windows × winsize × num_channels]."""
    n_samples = x.shape[0]
    if n_samples < winsize:
        return np.zeros((0, winsize, x.shape[1]), dtype=x.dtype)
    idx = np.arange(0, n_samples - winsize + 1, wininc)
    windows = np.stack([x[i:i + winsize] for i in idx])
    return windows


def get_category_from_emgname(emg_name):
    """Map an EMG filename (without the `PosX_` prefix) to a category id.

    This mirrors the original `getCategory` mapping used in the project.
    """
    prefix = emg_name[:9]

    if prefix == "WristFlex":
        return 1
    elif prefix == "WristExte":
        return 2
    elif prefix == "WristPron":
        return 3
    elif prefix == "WristSupi":
        return 4
    elif prefix == "ObjectGri":
        return 5
    elif prefix == "PichGrip_":
        return 6
    elif prefix == "HandOpen_":
        return 7
    elif prefix == "HandRest_":
        return 8
    else:
        raise ValueError(f"Unknown category prefix in: {emg_name}")


def load_emg_txt(path):
    """Load whitespace-separated floats from a text file into a numpy array."""
    return np.loadtxt(path)


def process_subject(subject_dir, out_path=None, winsize=400, wininc=100, positions=None, limit=0, extractor=extract_khushaba_features):
    """Process EMG `.txt` files for a subject and save extracted features.

    extractor : callable(window) -> 1D feature vector
    """
    subject_dir = Path(subject_dir)
    if positions:
        positions = set(positions)

    file_paths = sorted(subject_dir.glob("Pos*_*.txt"))

    X_list = []
    y_list = []
    src_files = []
    src_positions = []

    # group by position to optionally enforce per-position limit
    from collections import defaultdict
    files_by_pos = defaultdict(list)
    for p in file_paths:
        name = p.name
        pos, rest = name.split('_', 1)
        if positions and pos not in positions:
            continue
        files_by_pos[pos].append((p, rest))

    for pos in sorted(files_by_pos.keys()):
        items = files_by_pos[pos]
        if limit and limit > 0:
            items = items[:limit]

        for p, emg_name in items:
            try:
                category = get_category_from_emgname(emg_name)
            except ValueError as e:
                print(f"Skipping {p.name}: {e}")
                continue

            x = load_emg_txt(p)
            if x.ndim == 1:
                x = x[:, None]
            x = x[:, :7]

            windows = sliding_window(x, winsize, wininc)
            if windows.shape[0] == 0:
                continue

            for w in windows:
                feats = extractor(w)
                X_list.append(feats)
                y_list.append(category)
                src_files.append(p.name)
                src_positions.append(pos)

    if len(X_list) == 0:
        raise RuntimeError("No features extracted — check paths/positions/winsize")

    X = np.vstack([f.reshape(1, -1) if f.ndim == 1 else f for f in X_list])
    y = np.array(y_list, dtype=np.int64)

    if out_path is None:
        out_path = f"{subject_dir.name}_features.npz"

    np.savez_compressed(out_path, X=X, y=y, files=np.array(src_files), positions=np.array(src_positions))
    print(f"Saved features to: {out_path}  (X.shape={X.shape}, y.shape={y.shape})")


def parse_args():
    p = argparse.ArgumentParser(description="Extract features from EMG text files and save as .npz")
    p.add_argument("subject", help="Subject folder (e.g. S1_Male)")
    p.add_argument("--dataset", help="Dataset name (e.g. 'rami'). If provided, output paths are resolved from config.yaml")
    p.add_argument("--out", help="Output .npz file path (overrides --dataset output dir)")
    p.add_argument("--out-dir", help="Output directory to collect results (created if missing)")
    p.add_argument("--winsize", type=int, default=400)
    p.add_argument("--wininc", type=int, default=100)
    p.add_argument("--positions", help="Comma-separated positions to include (e.g. Pos1,Pos2)")
    p.add_argument("--limit", type=int, default=0, help="Limit files per position (0 = no limit)")
    p.add_argument("--feat", choices=["khushaba", "time"], default="khushaba", help="Feature set to extract: 'khushaba' (spectral) or 'time' (MAV,WL,ZC,SSC)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    positions = args.positions.split(',') if args.positions else None
    out_path = args.out
    
    # Resolve output directory from dataset config or explicit args
    out_dir = None
    if args.dataset:
        config = load_dataset_config(args.dataset)
        out_dir = config["_resolved"]["features_dir"]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path:
            out_p = Path(out_path)
            out_path = str(out_dir / out_p.name)
        else:
            # Default filename based on feature type
            suffix = "_features_khushaba.npz" if args.feat == "khushaba" else "_features_time.npz"
            out_path = str(out_dir / f"{Path(args.subject).name}{suffix}")

    # choose feature extractor
    if args.feat == 'khushaba':
        extractor = extract_khushaba_features
    else:
        extractor = extract_time_features

    process_subject(args.subject, out_path=out_path, winsize=args.winsize, wininc=args.wininc, positions=positions, limit=args.limit, extractor=extractor)
