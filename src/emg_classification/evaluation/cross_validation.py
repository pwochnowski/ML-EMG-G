"""Cross-validation utilities for EMG classification.

Provides LOSO (Leave-One-Subject-Out) and within-subject cross-validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import time

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

from ..models.factory import build_model
from ..preprocessing.normalizers import get_normalizer


logger = logging.getLogger(__name__)


def load_and_pool_subjects(
    results_dir: Union[Path, str],
    pattern: str = "*_features_combined.npz",
    subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and pool feature files from multiple subjects.
    
    Args:
        results_dir: Directory containing feature files.
        pattern: Glob pattern for feature files.
        subjects: Optional list of subjects to include.
        
    Returns:
        Tuple of (X, y, groups, positions) arrays.
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(pattern))
    
    Xs, ys, groups, positions = [], [], [], []
    
    for path in files:
        # Extract subject name from filename
        name = path.name
        if "_features_combined.npz" in name:
            subj = name.split("_features_combined.npz")[0]
        elif "_features.npz" in name:
            subj = name.split("_features.npz")[0]
        else:
            subj = name.split(".")[0]
        
        # Filter by subjects if specified
        if subjects is not None and subj not in subjects:
            continue
        
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        n = X.shape[0]
        
        Xs.append(X)
        ys.append(y)
        groups.append(np.array([subj] * n, dtype=object))
        
        # Load positions if available
        if "positions" in data:
            pos = data["positions"]
            if pos.shape[0] != n:
                pos = np.array(["unknown"] * n)
        else:
            pos = np.array(["unknown"] * n)
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


def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: List[str],
    normalizer: str = "standard",
    feature_selection: str = "none",
    k_best: Optional[int] = None,
    pca_components: Optional[int] = None,
    calibration_samples: int = 0,
    save_models_dir: Optional[Path] = None,
    return_predictions: bool = False,
) -> Dict:
    """Run Leave-One-Subject-Out cross-validation.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Labels of shape (n_samples,).
        groups: Subject identifiers of shape (n_samples,).
        model_names: List of model names to evaluate.
        normalizer: Normalization strategy.
        feature_selection: Feature selection method ('none', 'kbest', 'pca').
        k_best: Number of features for SelectKBest.
        pca_components: Number of PCA components.
        calibration_samples: Samples per class from test subject for calibration.
        save_models_dir: Directory to save trained models.
        return_predictions: Whether to return per-fold predictions.
        
    Returns:
        Dictionary with results for each model.
    """
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X, y, groups)
    results = {}
    
    for model_name in model_names:
        logger.info(f"Running LOSO CV for model '{model_name}' ({n_folds} folds)")
        
        fold_results = []
        all_predictions = [] if return_predictions else None
        
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            test_subject = groups[test_idx[0]]
            
            # Calibration: move some samples from test to train
            if calibration_samples > 0:
                calib_idx, remaining_idx = _select_calibration_samples(
                    y_te, calibration_samples, fold_idx
                )
                X_tr = np.vstack([X_tr, X_te[calib_idx]])
                y_tr = np.concatenate([y_tr, y_te[calib_idx]])
                X_te = X_te[remaining_idx]
                y_te = y_te[remaining_idx]
            
            # Build pipeline
            clf = _build_pipeline(
                model_name, normalizer, feature_selection, k_best, pca_components
            )
            
            # Train and evaluate
            t_start = time.time()
            clf.fit(X_tr, y_tr)
            t_fit = time.time() - t_start
            
            t_start = time.time()
            y_pred = clf.predict(X_te)
            t_pred = time.time() - t_start
            
            acc = accuracy_score(y_te, y_pred)
            
            fold_results.append({
                "fold": fold_idx,
                "subject": test_subject,
                "accuracy": acc,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "fit_time": t_fit,
                "predict_time": t_pred,
            })
            
            logger.info(
                f"  Fold {fold_idx + 1}/{n_folds} ({test_subject}): "
                f"acc={acc:.4f} (fit: {t_fit:.1f}s)"
            )
            
            if return_predictions:
                all_predictions.append({
                    "subject": test_subject,
                    "y_true": y_te,
                    "y_pred": y_pred,
                })
            
            # Save model
            if save_models_dir:
                save_models_dir.mkdir(parents=True, exist_ok=True)
                fname = save_models_dir / f"{model_name}_fold{fold_idx}.joblib"
                joblib.dump(clf, fname)
        
        # Aggregate results
        accuracies = [r["accuracy"] for r in fold_results]
        results[model_name] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "fold_results": fold_results,
            "predictions": all_predictions,
        }
        
        logger.info(
            f"Model '{model_name}': {results[model_name]['mean_accuracy']:.4f} "
            f"± {results[model_name]['std_accuracy']:.4f}"
        )
    
    return results


def run_within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: List[str],
    n_folds: int = 5,
    normalizer: str = "standard",
    feature_selection: str = "none",
    k_best: Optional[int] = None,
    pca_components: Optional[int] = None,
) -> Dict:
    """Run within-subject K-fold cross-validation.
    
    For each subject, runs stratified K-fold CV on their data alone.
    
    Args:
        X: Feature matrix.
        y: Labels.
        groups: Subject identifiers.
        model_names: List of model names.
        n_folds: Number of CV folds per subject.
        normalizer: Normalization strategy.
        feature_selection: Feature selection method.
        k_best: Number of features for SelectKBest.
        pca_components: Number of PCA components.
        
    Returns:
        Dictionary with results for each model and subject.
    """
    unique_subjects = np.unique(groups)
    results = {}
    
    for model_name in model_names:
        subject_results = {}
        
        for subj in unique_subjects:
            mask = groups == subj
            X_subj = X[mask]
            y_subj = y[mask]
            
            # Check if we have enough samples
            unique_classes, class_counts = np.unique(y_subj, return_counts=True)
            min_class_count = class_counts.min()
            actual_folds = min(n_folds, min_class_count)
            
            if actual_folds < 2:
                logger.warning(
                    f"Subject {subj}: Not enough samples for CV (min={min_class_count})"
                )
                continue
            
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            
            fold_accs = []
            for train_idx, test_idx in skf.split(X_subj, y_subj):
                X_tr, X_te = X_subj[train_idx], X_subj[test_idx]
                y_tr, y_te = y_subj[train_idx], y_subj[test_idx]
                
                clf = _build_pipeline(
                    model_name, normalizer, feature_selection,
                    min(k_best, X_subj.shape[1]) if k_best else None,
                    min(pca_components, X_subj.shape[1]) if pca_components else None,
                )
                
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                fold_accs.append(accuracy_score(y_te, y_pred))
            
            subject_results[subj] = {
                "mean_accuracy": float(np.mean(fold_accs)),
                "std_accuracy": float(np.std(fold_accs)),
                "n_folds": actual_folds,
            }
        
        # Aggregate across subjects
        all_means = [r["mean_accuracy"] for r in subject_results.values()]
        results[model_name] = {
            "overall_mean": float(np.mean(all_means)) if all_means else 0.0,
            "overall_std": float(np.std(all_means)) if all_means else 0.0,
            "subject_results": subject_results,
        }
    
    return results


def _build_pipeline(
    model_name: str,
    normalizer: str,
    feature_selection: str,
    k_best: Optional[int],
    pca_components: Optional[int],
):
    """Build a classification pipeline with optional feature selection."""
    # Get base model (without scaler since we use our normalizer)
    base_model = build_model(model_name, include_scaler=False)
    
    # Get normalizer
    norm = get_normalizer(normalizer)
    
    # Build pipeline components
    steps = [norm]
    
    if feature_selection == "kbest" and k_best:
        steps.append(SelectKBest(score_func=f_classif, k=k_best))
    elif feature_selection == "pca" and pca_components:
        steps.append(PCA(n_components=pca_components))
    
    steps.append(base_model)
    
    return make_pipeline(*steps)


def _select_calibration_samples(
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select calibration samples from test set.
    
    Returns indices for calibration and remaining test samples.
    """
    rng = np.random.RandomState(42 + seed)
    
    calib_idx = []
    remaining_idx = []
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        n_calib = min(n_samples, len(cls_indices) - 1)  # Keep at least 1 for testing
        
        if n_calib > 0:
            selected = rng.choice(cls_indices, size=n_calib, replace=False)
            calib_idx.extend(selected)
            remaining_idx.extend([i for i in cls_indices if i not in selected])
        else:
            remaining_idx.extend(cls_indices)
    
    return np.array(calib_idx), np.array(remaining_idx)


def save_results_csv(
    results: Dict,
    output_path: Union[Path, str],
    cv_type: str = "loso",
) -> None:
    """Save cross-validation results to CSV.
    
    Args:
        results: Results dictionary from run_loso_cv or run_within_subject_cv.
        output_path: Path to save CSV.
        cv_type: Type of CV ('loso' or 'within_subject').
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        if cv_type == "loso":
            writer = csv.writer(f)
            writer.writerow(["model", "mean_accuracy", "std_accuracy", "n_folds"])
            for model, data in results.items():
                writer.writerow([
                    model,
                    f"{data['mean_accuracy']:.4f}",
                    f"{data['std_accuracy']:.4f}",
                    len(data["fold_results"]),
                ])
        else:
            writer = csv.writer(f)
            writer.writerow(["model", "subject", "mean_accuracy", "std_accuracy", "n_folds"])
            for model, data in results.items():
                for subj, subj_data in data["subject_results"].items():
                    writer.writerow([
                        model,
                        subj,
                        f"{subj_data['mean_accuracy']:.4f}",
                        f"{subj_data['std_accuracy']:.4f}",
                        subj_data["n_folds"],
                    ])
    
    logger.info(f"Saved results to {output_path}")
