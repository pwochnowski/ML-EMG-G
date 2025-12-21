"""Combine two .npz feature files (spectral + time) into one concatenated dataset.

Usage:
  uv run combine_features.py --spec datasets/rami/features/S1_Male_features_khushaba.npz \
      --time datasets/rami/features/S1_Male_features_time.npz --out datasets/rami/features/S1_Male_features_combined.npz
  
  # Or use --dataset to auto-resolve paths:
  uv run combine_features.py --dataset rami --subject S1_Male
"""
from pathlib import Path
import argparse
import numpy as np
import yaml


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', help='Spectral features .npz (required unless --dataset is used)')
    p.add_argument('--time', help='Time features .npz (required unless --dataset is used)')
    p.add_argument('--out', help='Output combined .npz (required unless --dataset is used)')
    p.add_argument('--dataset', help='Dataset name (e.g. "rami"). Auto-resolves paths from config.yaml')
    p.add_argument('--subject', help='Subject name (e.g. "S1_Male"). Required when using --dataset')
    return p.parse_args()


def main():
    args = parse_args()
    
    # Resolve paths from dataset config or explicit arguments
    if args.dataset:
        if not args.subject:
            raise SystemExit("--subject is required when using --dataset")
        config = load_dataset_config(args.dataset)
        features_dir = config["_resolved"]["features_dir"]
        spec_path = features_dir / f"{args.subject}_features_khushaba.npz"
        time_path = features_dir / f"{args.subject}_features_time.npz"
        out_path = features_dir / f"{args.subject}_features_combined.npz"
    else:
        if not args.spec or not args.time or not args.out:
            raise SystemExit("Either use --dataset/--subject or provide --spec, --time, and --out")
        spec_path = Path(args.spec)
        time_path = Path(args.time)
        out_path = Path(args.out)
    
    spec = np.load(spec_path, allow_pickle=True)
    time = np.load(time_path, allow_pickle=True)

    Xs = spec['X']
    ys = spec['y']
    files_s = spec.get('files')
    pos_s = spec.get('positions')

    Xt = time['X']
    yt = time['y']
    files_t = time.get('files')
    pos_t = time.get('positions')

    if Xs.shape[0] != Xt.shape[0]:
        raise SystemExit(f"Sample count mismatch: spec {Xs.shape[0]} vs time {Xt.shape[0]}")

    if not np.array_equal(ys, yt):
        # try to detect common ordering via files+positions
        if files_s is not None and files_t is not None and pos_s is not None and pos_t is not None:
            keys_s = np.core.defchararray.add(files_s.astype(str), pos_s.astype(str)) # type: ignore
            keys_t = np.core.defchararray.add(files_t.astype(str), pos_t.astype(str)) # type: ignore
            if np.array_equal(keys_s, keys_t):
                pass
            else:
                raise SystemExit('Labels or sample order do not match between feature files')
        else:
            raise SystemExit('Labels do not match between feature files')

    X_comb = np.hstack([Xs, Xt])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X_comb, y=ys, files=files_s, positions=pos_s)
    print(f"Saved combined features to: {out_path} (X.shape={X_comb.shape})")


if __name__ == '__main__':
    main()
