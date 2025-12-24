"""Leave-One-Subject-Out (LOSO) training/evaluation helper.

This script looks for per-subject combined feature files in a directory
(`datasets/{dataset}/features/*_features_combined.npz` by default), pools them into a single
dataset (adding a `subject` group label), and runs Leave-One-Group-Out
evaluation (one subject left out per fold) for the requested models.

It reports per-model mean/std accuracy across folds and writes a CSV
summary to `datasets/{dataset}/training/loso_summary.csv` by default. Optionally saves per-fold
models to a directory.

Feature Sets:
    Feature extraction is done separately (see extract_features.py or combine_features.py).
    Different feature sets are defined in src/emg_classification/config/features.yaml.
    Use --list-features to see available feature set names.
"""

from typing import Optional
from pathlib import Path
import argparse
import numpy as np
import logging
import time
import yaml
from datetime import datetime
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
from models import build_model
from sklearn.metrics import accuracy_score, f1_score
import joblib
import csv
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from domain_adaptation import (SubjectNormalizer, PercentileNormalizer, 
                                ChannelNormalizer, SubjectAdaptiveScaler)

# Feature set configuration (for --list-features)
from src.emg_classification.config.schema import list_feature_sets


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
    
    # Get feature_subdir from config (default: combined/all)
    feature_subdir = config.get("feature_subdir", "combined/all")
    
    config["_resolved"] = {
        "features_dir": dataset_root / output.get("features_dir", "features"),
        "training_dir": dataset_root / output.get("training_dir", "training"),
        "tuning_dir": dataset_root / output.get("tuning_dir", "tuning"),
        "models_dir": dataset_root / output.get("models_dir", "models"),
        "analysis_dir": dataset_root / output.get("analysis_dir", "analysis"),
        "data_dir": dataset_root / "data",
        "feature_subdir": feature_subdir,
    }
    
    # Get subject list from config
    dataset_cfg = config.get("dataset", {})
    subjects = dataset_cfg.get("subjects", None)
    if subjects:
        if isinstance(subjects, dict):
            # Mapping format (rami): {folder: id} -> list of ids
            config["_resolved"]["subjects"] = list(subjects.values())
            config["_resolved"]["subject_mapping"] = subjects
        else:
            # List format (db1): [id1, id2, ...]
            config["_resolved"]["subjects"] = subjects
            config["_resolved"]["subject_mapping"] = None
    else:
        config["_resolved"]["subjects"] = None
        config["_resolved"]["subject_mapping"] = None
    
    return config


# build_model is provided by the shared `models` module

def get_normalizer(name: str):
    """Get a normalizer by name."""
    normalizers = {
        'standard': StandardScaler(),
        'percentile': PercentileNormalizer(),
        'channel': ChannelNormalizer(),
        'adaptive': SubjectAdaptiveScaler(),
    }
    if name not in normalizers:
        raise ValueError(f'Unknown normalizer: {name}. Available: {list(normalizers.keys())}')
    return normalizers[name]


def generate_run_name(models: list[str], accuracy: float) -> str:
    """Generate run folder name from models and accuracy.
    
    Format: {model(s)}_acc{accuracy:.3f}
    Examples:
        - et-tuned_acc0.847
        - et-tuned_lda_acc0.823
    """
    model_str = '_'.join(models)
    return f"{model_str}_acc{accuracy:.3f}"


def resolve_run_path(base_dir: Path, run_name: str) -> Path:
    """Resolve run path, handling collisions with timestamp suffix.
    
    If the path already exists, appends _dd_hhmm timestamp.
    """
    run_path = base_dir / run_name
    if run_path.exists():
        timestamp = datetime.now().strftime("%d_%H%M")
        run_path = base_dir / f"{run_name}_{timestamp}"
    return run_path


def save_run_config(run_dir: Path, args: argparse.Namespace, results: list, 
                    dataset_info: dict = None):
    """Save configuration and metadata for a training run.
    
    Args:
        run_dir: Directory to save config
        args: Parsed command-line arguments
        results: List of (model, accuracy, std, macro_f1, n_folds) tuples
        dataset_info: Optional dict with dataset metadata
    """
    config = {
        'feature_set': args.feature_set,
        'dataset': args.dataset,
        'models': [m.strip() for m in args.models.split(',')],
        'normalizer': args.normalizer,
        'feat_select': args.feat_select,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add optional parameters if set
    if args.k:
        config['k'] = args.k
    if args.pca_n:
        config['pca_n'] = args.pca_n
    if args.calibration:
        config['calibration_samples'] = args.calibration
    if args.subsample:
        config['subsample_frac'] = args.subsample
    if args.subsample_n:
        config['subsample_n'] = args.subsample_n
    if args.labels:
        config['labels'] = args.labels
    if args.feature_subdir:
        config['feature_subdir'] = args.feature_subdir
    
    # Add dataset info if provided
    if dataset_info:
        config['dataset_info'] = dataset_info
    
    # Save config
    config_path = run_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Saved run config to {config_path}")


def subsample_data(X, y, groups, positions, subsample_frac: Optional[float] = None, 
                   subsample_n: Optional[int] = None, random_state: int = 42):
    """
    Subsample the dataset while maintaining stratification by class and subject.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Subject groupings
        positions: Position labels
        subsample_frac: Fraction of data to keep (0.0-1.0)
        subsample_n: Absolute number of samples to keep per subject (alternative to frac)
        random_state: Random seed for reproducibility
    
    Returns:
        Subsampled X, y, groups, positions
    """
    if subsample_frac is None and subsample_n is None:
        return X, y, groups, positions
    
    unique_subjects = np.unique(groups)
    X_sub, y_sub, groups_sub, positions_sub = [], [], [], []
    
    np.random.seed(random_state)
    
    for subj in unique_subjects:
        mask = (groups == subj)
        X_s = X[mask]
        y_s = y[mask]
        pos_s = positions[mask]
        
        n_samples = X_s.shape[0]
        
        if subsample_n is not None:
            # Keep exactly N samples per subject (or all if fewer)
            n_keep = min(subsample_n, n_samples)
        else:
            # Keep a fraction of samples
            n_keep = max(1, int(n_samples * subsample_frac))
        
        if n_keep >= n_samples:
            # Keep all samples for this subject
            X_sub.append(X_s)
            y_sub.append(y_s)
            groups_sub.append(np.array([subj] * n_samples, dtype=object))
            positions_sub.append(pos_s)
        else:
            # Stratified subsampling by class
            unique_classes = np.unique(y_s)
            indices_to_keep = []
            
            # Calculate samples per class (proportional)
            class_counts = {c: np.sum(y_s == c) for c in unique_classes}
            total = sum(class_counts.values())
            
            for cls in unique_classes:
                cls_indices = np.where(y_s == cls)[0]
                # Proportional allocation
                n_cls_keep = max(1, int(n_keep * class_counts[cls] / total))
                n_cls_keep = min(n_cls_keep, len(cls_indices))
                
                selected = np.random.choice(cls_indices, size=n_cls_keep, replace=False)
                indices_to_keep.extend(selected)
            
            indices_to_keep = np.array(indices_to_keep)
            X_sub.append(X_s[indices_to_keep])
            y_sub.append(y_s[indices_to_keep])
            groups_sub.append(np.array([subj] * len(indices_to_keep), dtype=object))
            positions_sub.append(pos_s[indices_to_keep])
    
    X_out = np.vstack(X_sub)
    y_out = np.concatenate(y_sub)
    groups_out = np.concatenate(groups_sub)
    positions_out = np.concatenate(positions_sub)
    
    return X_out, y_out, groups_out, positions_out


def collect_subject_files(results_dir: Path, pattern: str, subject_list: Optional[list] = None):
    """Collect subject feature files from results directory.
    
    Args:
        results_dir: Directory containing feature files
        pattern: Glob pattern for feature files (e.g., '*.npz')
        subject_list: Optional list of subject IDs to look for. If provided,
                      files are matched by name (e.g., 's01.npz'). If None,
                      all matching files are collected.
    
    Returns:
        List of (subject_id, file_path) tuples
    """
    if subject_list:
        # Look for specific subjects from config
        subjects = []
        for subj in subject_list:
            # Try exact match first (new format: s01.npz)
            path = results_dir / f"{subj}.npz"
            if path.exists():
                subjects.append((subj, path))
            else:
                # Fall back to legacy format: s01_features_combined.npz
                legacy_path = results_dir / f"{subj}_features_combined.npz"
                if legacy_path.exists():
                    subjects.append((subj, legacy_path))
        return subjects
    
    # No subject list provided - discover from files
    files = sorted(results_dir.glob(pattern))
    subjects = []
    for f in files:
        name = f.name
        # Handle both new format (s01.npz) and legacy format (s01_features_combined.npz)
        if '_features_combined.npz' in name:
            subj = name.split('_features_combined.npz')[0]
        elif '_features.npz' in name:
            subj = name.split('_features.npz')[0]
        elif name.endswith('.npz'):
            subj = name.replace('.npz', '')
        else:
            subj = name.split('.')[0]
        subjects.append((subj, f))
    return subjects


def load_and_pool(subject_files):
    Xs = []
    ys = []
    groups = []
    positions = []
    for subj, path in subject_files:
        data = np.load(path, allow_pickle=True)
        X = data['X']
        y = data['y']
        n = X.shape[0]
        Xs.append(X)
        ys.append(y)
        groups.append(np.array([subj] * n, dtype=object))
        # optional 'positions' array saved by the feature extractor
        if 'positions' in data:
            pos = data['positions']
            if pos.shape[0] != n:
                # fallback: repeat a single position label
                pos = np.array([str(data.get('position', 'unknown'))] * n)
        else:
            pos = np.array([str(data.get('position', 'unknown'))] * n)
        positions.append(pos)

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    groups_all = np.concatenate(groups)
    positions_all = np.concatenate(positions)
    
    # Encode labels to start from 0 (required for XGBoost etc.)
    unique_labels = np.unique(y_all)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_all = np.array([label_map[yi] for yi in y_all])
    
    return X_all, y_all, groups_all, positions_all


def run_training(X, y, groups, models, use_cv: bool = True,
                 save_models_dir: Optional[Path] = None, out_csv: Optional[Path] = None,
                 feat_select: str = 'none', k: Optional[int] = None, pca_n: Optional[int] = None,
                 normalizer: str = 'standard', calibration_samples: int = 0):
    """
    Unified training function that handles both CV and no-CV modes.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Subject groupings (used for LOSO CV)
        models: List of model names to evaluate
        use_cv: If True, run LOSO cross-validation; if False, train on full dataset
        save_models_dir: Directory to save models (optional)
        out_csv: Output CSV file path (optional)
        feat_select: Feature selection method ('none', 'kbest', 'pca')
        k: Number of features for SelectKBest
        pca_n: Number of PCA components
        normalizer: Normalization strategy ('standard', 'percentile', 'channel', 'adaptive')
        calibration_samples: Number of samples per class from test subject to add to training (0 = no calibration)
    
    Returns:
        List of tuples: (model_name, mean_acc, std_acc, macro_f1, n_folds)
    """
    results = []
    logo = LeaveOneGroupOut() if use_cv else None

    for model_name in models:
        # Build base estimator and optionally insert a selector/pca step
        base_pipe = build_model(model_name)
        try:
            estimator = base_pipe.steps[-1][1]
        except Exception:
            estimator = base_pipe

        # Create selector if requested
        selector = None
        if feat_select == 'kbest':
            if not k:
                raise ValueError('k must be provided when --feat-select kbest')
            selector = SelectKBest(score_func=f_classif, k=int(k))
        elif feat_select == 'pca':
            if not pca_n:
                raise ValueError('pca-n must be provided when --feat-select pca')
            selector = PCA(n_components=int(pca_n))

        # Get normalizer
        norm = get_normalizer(normalizer)

        # Build final pipeline
        if selector is not None:
            clf = make_pipeline(norm, selector, estimator)
        else:
            # Replace the StandardScaler in base_pipe with our normalizer
            clf = make_pipeline(norm, estimator)

        if use_cv:
            # LOSO Cross-Validation
            fold_acc = []
            fold_f1 = []
            n_folds = sum(1 for _ in logo.split(X, y, groups))
            logging.info("Running model '%s' with %d LOSO folds (feat_select=%s)", model_name, n_folds, feat_select)
            logging.info("Dataset size: %d samples, %d features", X.shape[0], X.shape[1])
            
            for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Calibration: move some samples from test to train
                if calibration_samples > 0:
                    calib_idx = []
                    remaining_idx = []
                    for cls in np.unique(y_te):
                        cls_indices = np.where(y_te == cls)[0]
                        n_calib = min(calibration_samples, len(cls_indices) - 1)  # Keep at least 1 for testing
                        if n_calib > 0:
                            np.random.seed(42 + fold_idx)  # Reproducible per fold
                            selected = np.random.choice(cls_indices, size=n_calib, replace=False)
                            calib_idx.extend(selected)
                            remaining_idx.extend([i for i in cls_indices if i not in selected])
                        else:
                            remaining_idx.extend(cls_indices)
                    
                    # Add calibration samples to training
                    calib_idx = np.array(calib_idx)
                    remaining_idx = np.array(remaining_idx)
                    X_tr = np.vstack([X_tr, X_te[calib_idx]])
                    y_tr = np.concatenate([y_tr, y_te[calib_idx]])
                    X_te = X_te[remaining_idx]
                    y_te = y_te[remaining_idx]
                    logging.info("%s - fold %d/%d: using %d calibration samples from test subject", 
                                model_name, fold_idx + 1, n_folds, len(calib_idx))

                logging.info("%s - fold %d/%d: training on %d samples...", model_name, fold_idx + 1, n_folds, len(X_tr))
                t_start = time.time()
                clf.fit(X_tr, y_tr)
                t_fit = time.time() - t_start
                
                t_start = time.time()
                y_pred = clf.predict(X_te)
                t_pred = time.time() - t_start
                
                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average='macro')
                fold_acc.append(acc)
                fold_f1.append(f1)

                logging.info("%s - fold %d/%d: acc=%.4f, f1=%.4f (fit: %.1fs, predict: %.1fs)", 
                            model_name, fold_idx + 1, n_folds, acc, f1, t_fit, t_pred)

                if save_models_dir:
                    save_models_dir.mkdir(parents=True, exist_ok=True)
                    fname = save_models_dir / f"{model_name}_fold{fold_idx}.joblib"
                    joblib.dump(clf, fname)
                    logging.debug("Saved model to %s", fname)

            mean_acc = float(np.mean(fold_acc))
            std_acc = float(np.std(fold_acc))
            mean_f1 = float(np.mean(fold_f1))
            results.append((model_name, mean_acc, std_acc, mean_f1, len(fold_acc)))
            print(f"Model: {model_name}  LOSO accuracy: {mean_acc:.4f} ± {std_acc:.4f}  macro-F1: {mean_f1:.4f}  (folds={len(fold_acc)})")

        else:
            # No CV - train on full dataset
            logging.info("Training model '%s' on full dataset (no CV), feat_select=%s", model_name, feat_select)
            logging.info("Dataset size: %d samples, %d features", X.shape[0], X.shape[1])
            
            t_start = time.time()
            clf.fit(X, y)
            t_fit = time.time() - t_start
            logging.info("Fit completed in %.1f seconds", t_fit)
            
            t_start = time.time()
            y_pred = clf.predict(X)
            t_pred = time.time() - t_start
            logging.info("Predict completed in %.1f seconds", t_pred)
            
            train_acc = accuracy_score(y, y_pred)
            train_f1 = f1_score(y, y_pred, average='macro')

            results.append((model_name, train_acc, 0.0, train_f1, 1))
            print(f"Model: {model_name}  Training accuracy: {train_acc:.4f}  macro-F1: {train_f1:.4f}  (fit: {t_fit:.1f}s, predict: {t_pred:.1f}s)")

            if save_models_dir:
                save_models_dir.mkdir(parents=True, exist_ok=True)
                fname = save_models_dir / f"{model_name}_full.joblib"
                joblib.dump(clf, fname)
                logging.debug("Saved model to %s", fname)

    # Write results to CSV
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            header = ['model', 'mean_acc' if use_cv else 'train_acc', 'std_acc', 'macro_f1', 'n_folds']
            writer.writerow(header)
            for r in results:
                writer.writerow(r)
        print(f"Saved summary to: {out_csv}")

    return results


def run_within_subject_cv(X, y, groups, models, n_folds: int = 5,
                          save_models_dir: Optional[Path] = None, out_csv: Optional[Path] = None,
                          feat_select: str = 'none', k: Optional[int] = None, pca_n: Optional[int] = None):
    """
    Within-subject cross-validation: for each subject, run K-Fold CV on their data alone.
    Reports per-subject accuracy and overall mean across subjects.
    """
    unique_subjects = np.unique(groups)
    all_results = []  # (subject, model, mean_acc, std_acc, n_folds)
    
    for subj in unique_subjects:
        mask = (groups == subj)
        X_subj = X[mask]
        y_subj = y[mask]
        
        print(f"\n{'='*60}")
        print(f"Subject: {subj} ({X_subj.shape[0]} samples, {X_subj.shape[1]} features)")
        print(f"{'='*60}")
        
        # Check if we have enough samples per class for stratified k-fold
        unique_classes, class_counts = np.unique(y_subj, return_counts=True)
        min_class_count = class_counts.min()
        actual_folds = min(n_folds, min_class_count)
        
        if actual_folds < 2:
            logging.warning(f"Subject {subj}: Not enough samples per class for CV (min={min_class_count}). Skipping.")
            continue
        
        if actual_folds < n_folds:
            logging.warning(f"Subject {subj}: Reduced folds from {n_folds} to {actual_folds} (min class count={min_class_count})")
        
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        
        for model_name in models:
            # Build base estimator
            base_pipe = build_model(model_name)
            try:
                estimator = base_pipe.steps[-1][1]
            except Exception:
                estimator = base_pipe
            
            # Create selector if requested
            selector = None
            if feat_select == 'kbest':
                if not k:
                    raise ValueError('k must be provided when --feat-select kbest')
                selector = SelectKBest(score_func=f_classif, k=min(int(k), X_subj.shape[1]))
            elif feat_select == 'pca':
                if not pca_n:
                    raise ValueError('pca-n must be provided when --feat-select pca')
                selector = PCA(n_components=min(int(pca_n), X_subj.shape[1], X_subj.shape[0]))
            
            # Build final pipeline
            if selector is not None:
                clf = make_pipeline(StandardScaler(), selector, estimator)
            else:
                clf = base_pipe
            
            fold_acc = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subj, y_subj)):
                X_tr, X_te = X_subj[train_idx], X_subj[test_idx]
                y_tr, y_te = y_subj[train_idx], y_subj[test_idx]
                
                t_start = time.time()
                clf.fit(X_tr, y_tr)
                t_fit = time.time() - t_start
                
                y_pred = clf.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                fold_acc.append(acc)
                
                logging.debug(f"{subj}/{model_name} fold {fold_idx+1}/{actual_folds}: acc={acc:.4f} (fit: {t_fit:.1f}s)")
            
            mean_acc = float(np.mean(fold_acc))
            std_acc = float(np.std(fold_acc))
            all_results.append((subj, model_name, mean_acc, std_acc, actual_folds))
            print(f"  {model_name}: {mean_acc:.4f} ± {std_acc:.4f} ({actual_folds} folds)")
            
            # Save model trained on full subject data
            if save_models_dir:
                save_models_dir.mkdir(parents=True, exist_ok=True)
                clf.fit(X_subj, y_subj)  # retrain on all subject data
                fname = save_models_dir / f"{model_name}_{subj}.joblib"
                joblib.dump(clf, fname)
    
    # Print summary across subjects
    print(f"\n{'='*60}")
    print("SUMMARY: Mean accuracy across subjects")
    print(f"{'='*60}")
    
    for model_name in models:
        model_results = [(s, m, a, std, n) for s, m, a, std, n in all_results if m == model_name]
        if model_results:
            accs = [a for _, _, a, _, _ in model_results]
            overall_mean = np.mean(accs)
            overall_std = np.std(accs)
            print(f"{model_name}: {overall_mean:.4f} ± {overall_std:.4f} (across {len(model_results)} subjects)")
    
    # Write results to CSV
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['subject', 'model', 'mean_acc', 'std_acc', 'n_folds'])
            for r in all_results:
                writer.writerow(r)
        print(f"\nSaved per-subject results to: {out_csv}")
    
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description='LOSO evaluation across subjects')
    p.add_argument('--dataset', help='Dataset name (e.g. "rami"). Auto-resolves paths from config.yaml')
    p.add_argument('--feature-set', 
                   help='Name of the feature set used (from features.yaml). Required for training runs.')
    p.add_argument('--results-dir', help='Directory containing per-subject combined .npz files (overrides --dataset)')
    p.add_argument('--feature-subdir', help='Feature subdirectory (e.g., "combined/all", "spectral/e1"). Overrides config default.')
    p.add_argument('--pattern', default='*.npz', help='Glob pattern for feature files (default: *.npz)')
    p.add_argument('--models', default='lda,svm', help='Comma-separated models to evaluate')
    p.add_argument('--save-models-dir', help='Directory to save per-fold models (optional)')
    p.add_argument('--out-csv', help='Output CSV file for LOSO summary (default: auto from --dataset)')
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    p.add_argument('--per-position', action='store_true', help='Run LOSO separately for each position found in pooled data')
    p.add_argument('--positions', help='Comma-separated list of positions to evaluate (default: all)')
    p.add_argument('--subjects', help='Comma-separated list of subjects to include (default: all). E.g. --subjects s01,s02')
    p.add_argument('--feat-select', default='none', choices=['none', 'kbest', 'pca'], help='Feature selection method to apply (per-fold)')
    p.add_argument('--k', type=int, default=50, help='Number of features for SelectKBest (used when --feat-select=kbest)')
    p.add_argument('--pca-n', type=int, default=50, help='Number of PCA components (used when --feat-select=pca)')
    p.add_argument('--no-cv', action='store_true', help='Train on full dataset without cross-validation (reports training accuracy)')
    p.add_argument('--within-subject', action='store_true', help='Run within-subject K-Fold CV instead of LOSO')
    p.add_argument('--n-folds', type=int, default=5, help='Number of folds for within-subject CV (default: 5)')
    p.add_argument('--subsample', type=float, help='Fraction of data to subsample (0.0-1.0). E.g. --subsample 0.1 keeps 10%% of data')
    p.add_argument('--subsample-n', type=int, help='Number of samples to keep per subject (alternative to --subsample)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for subsampling (default: 42)')
    p.add_argument('--normalizer', default='standard', choices=['standard', 'percentile', 'channel', 'adaptive'],
                   help='Normalization strategy for domain adaptation (default: standard)')
    p.add_argument('--calibration', type=int, default=0, 
                   help='Number of samples per class from test subject to use for calibration (default: 0 = no calibration)')
    p.add_argument('--labels', help='Filter to specific labels. Can be comma-separated (1,2,3) or range (13-20) or mix (1-12,30-52)')
    p.add_argument('--list-features', action='store_true', 
                   help='List available feature sets from features.yaml and exit')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Handle --list-features
    if args.list_features:
        print("Available feature sets (defined in src/emg_classification/config/features.yaml):")
        for name in list_feature_sets():
            print(f"  - {name}")
        print("\nUse these names when extracting features (e.g., in combine_features.py)")
        raise SystemExit(0)
    
    # Require --feature-set for actual training runs
    if not args.feature_set:
        raise SystemExit("Error: --feature-set is required for training runs. Use --list-features to see available options.")
    
    # Resolve paths from dataset config or explicit arguments
    subject_list = None
    dataset_config = None
    if args.dataset:
        dataset_config = load_dataset_config(args.dataset)
        
        # Determine feature_subdir (CLI arg overrides config)
        feature_subdir = args.feature_subdir or dataset_config["_resolved"]["feature_subdir"]
        
        # Build results directory: features_dir / feature_subdir
        results_dir = dataset_config["_resolved"]["features_dir"] / feature_subdir
        
        # Base training directory for output organization
        training_base_dir = dataset_config["_resolved"]["training_dir"]
        subject_list = dataset_config["_resolved"]["subjects"]
    else:
        results_dir = Path(args.results_dir) if args.results_dir else Path("results")
        training_base_dir = results_dir / "training"
    
    files = collect_subject_files(results_dir, args.pattern, subject_list)
    if not files:
        raise SystemExit(f'No files found matching pattern {args.pattern} in {results_dir}')

    # Filter by subjects if specified
    if args.subjects:
        requested_subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
        files = [(subj, path) for subj, path in files if subj in requested_subjects]
        if not files:
            raise SystemExit(f'No matching subjects found. Requested: {requested_subjects}')

    print(f'Found {len(files)} subject files. Subjects: {[s for s,_ in files]}')
    X, y, groups, positions = load_and_pool(files)
    print(f'Pooled dataset: X.shape={X.shape}, y.shape={y.shape}, subjects={len(np.unique(groups))}')

    # Filter by labels if specified
    if args.labels:
        label_set = set()
        for part in args.labels.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                label_set.update(range(start, end + 1))
            else:
                label_set.add(int(part))
        mask = np.isin(y, list(label_set))
        X, y, groups, positions = X[mask], y[mask], groups[mask], positions[mask]
        # Re-encode labels to be contiguous starting from 0
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[yi] for yi in y])
        print(f'After label filter ({args.labels}): X.shape={X.shape}, classes={len(unique_labels)}')

    # Apply subsampling if requested
    if args.subsample or args.subsample_n:
        X_orig_shape = X.shape
        X, y, groups, positions = subsample_data(
            X, y, groups, positions,
            subsample_frac=args.subsample,
            subsample_n=args.subsample_n,
            random_state=args.seed
        )
        print(f'After subsampling: X.shape={X.shape} (was {X_orig_shape}), reduction={1 - X.shape[0]/X_orig_shape[0]:.1%}')

    # prepare model list
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    use_cv = not args.no_cv

    # configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Logging level set to %s', args.log_level.upper())

    # Dataset info for config saving
    dataset_info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_subjects': len(np.unique(groups)),
        'n_classes': len(np.unique(y)),
        'subjects': [s for s, _ in files],
    }

    # Handle within-subject CV mode
    if args.within_subject:
        # Within-subject mode uses legacy output paths
        legacy_out_csv = training_base_dir / "loso_summary_within_subject.csv"
        out_csv_within = Path(args.out_csv) if args.out_csv else legacy_out_csv
        run_within_subject_cv(X, y, groups, models, n_folds=args.n_folds,
                              save_models_dir=None, out_csv=out_csv_within,
                              feat_select=args.feat_select, k=args.k, pca_n=args.pca_n)
    # handle per-position evaluation
    elif args.per_position:
        pos_arr = np.array(positions, dtype=str)
        all_positions = np.unique(pos_arr)
        if args.positions:
            requested = [p.strip() for p in args.positions.split(',') if p.strip()]
            positions_to_run = [p for p in requested if p in all_positions]
        else:
            positions_to_run = list(all_positions)

        if not positions_to_run:
            raise SystemExit('No matching positions found to evaluate')

        for pos in positions_to_run:
            logging.info("Starting position '%s'", pos)
            mask = (pos_arr == pos)
            X_pos = X[mask]
            y_pos = y[mask]
            groups_pos = groups[mask]
            unique_subjects = np.unique(groups_pos)
            logging.info("Position '%s': samples=%d, subjects=%d", pos, X_pos.shape[0], len(unique_subjects))
            
            if use_cv and len(unique_subjects) < 2:
                logging.warning("Skipping position '%s' because fewer than 2 subjects present (needed for CV)", pos)
                continue

            # Run training first to get results
            results = run_training(X_pos, y_pos, groups_pos, models, use_cv=use_cv,
                         save_models_dir=None, out_csv=None,
                         feat_select=args.feat_select, k=args.k, pca_n=args.pca_n,
                         normalizer=args.normalizer, calibration_samples=args.calibration)
            
            # Generate output path based on best accuracy
            if results:
                best_acc = max(r[1] for r in results)  # r[1] is mean_acc
                run_name = generate_run_name(models, best_acc)
                run_name = f"{run_name}_{pos}"  # Append position
                
                # Output structure: training/{feature_set}/{run_name}/
                feature_set_dir = training_base_dir / args.feature_set
                run_dir = resolve_run_path(feature_set_dir, run_name)
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # Save results CSV
                out_csv = run_dir / "results.csv"
                with open(out_csv, 'w', newline='') as fh:
                    writer = csv.writer(fh)
                    header = ['model', 'accuracy', 'std', 'macro_f1', 'n_folds']
                    writer.writerow(header)
                    for r in results:
                        writer.writerow(r)
                print(f"Saved results to: {out_csv}")
                
                # Save run config
                save_run_config(run_dir, args, results, dataset_info)
            
            logging.info("Finished position '%s'", pos)
    elif not args.within_subject:
        # Run on full dataset (LOSO or no-CV)
        results = run_training(X, y, groups, models, use_cv=use_cv,
                     save_models_dir=None, out_csv=None,
                     feat_select=args.feat_select, k=args.k, pca_n=args.pca_n,
                     normalizer=args.normalizer, calibration_samples=args.calibration)
        
        # Generate output path based on best accuracy
        if results:
            best_acc = max(r[1] for r in results)  # r[1] is mean_acc
            run_name = generate_run_name(models, best_acc)
            
            # Output structure: training/{feature_set}/{run_name}/
            feature_set_dir = training_base_dir / args.feature_set
            run_dir = resolve_run_path(feature_set_dir, run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results CSV
            out_csv = run_dir / "results.csv"
            with open(out_csv, 'w', newline='') as fh:
                writer = csv.writer(fh)
                header = ['model', 'accuracy', 'std', 'macro_f1', 'n_folds']
                writer.writerow(header)
                for r in results:
                    writer.writerow(r)
            print(f"Saved results to: {out_csv}")
            
            # Save run config
            save_run_config(run_dir, args, results, dataset_info)
            
            print(f"\n{'='*60}")
            print(f"Run output saved to: {run_dir}")
            print(f"{'='*60}")
