"""Shared utilities for loading dataset configurations and paths."""
from pathlib import Path
import yaml
import numpy as np


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


def load_features_for_tuning(features_dir: Path, subsample_frac: float = 0.05, 
                             random_state: int = 42, encode_labels: bool = False):
    """Load and subsample feature data for hyperparameter tuning.
    
    Args:
        features_dir: Path to directory containing *_features_combined.npz files
        subsample_frac: Fraction of data to keep (stratified by subject and label)
        random_state: Random seed for reproducibility
        encode_labels: Whether to re-encode labels to 0-indexed integers
        
    Returns:
        X, y, groups: Feature matrix, labels, and subject group labels
    """
    data_files = sorted(features_dir.glob('*_features_combined.npz'))
    if not data_files:
        raise FileNotFoundError(f"No *_features_combined.npz files found in {features_dir}")
    
    X_all, y_all, groups_all = [], [], []
    for f in data_files:
        data = np.load(f)
        subj = f.name.replace('_features_combined.npz', '')
        X_all.append(data['X'])
        y_all.append(data['y'])
        groups_all.append(np.array([subj] * len(data['X'])))
    
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    groups = np.concatenate(groups_all)
    
    # Optionally encode labels (needed for XGBoost, LightGBM)
    if encode_labels:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Subsample - STRATIFIED by subject AND label
    np.random.seed(random_state)
    idx = []
    for subj in np.unique(groups):
        subj_mask = groups == subj
        subj_idx = np.where(subj_mask)[0]
        subj_y = y[subj_mask]
        
        for label in np.unique(subj_y):
            label_idx = subj_idx[subj_y == label]
            n_keep = max(1, int(len(label_idx) * subsample_frac))
            idx.extend(np.random.choice(label_idx, size=n_keep, replace=False))
    
    idx = np.array(idx)
    return X[idx], y[idx], groups[idx]
