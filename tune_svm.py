"""Hyperparameter tuning script for SVM using cross-validation.

Usage:
    uv run python tune_svm.py --dataset rami
"""
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

from dataset_utils import load_dataset_config, load_features_for_tuning


def main():
    parser = argparse.ArgumentParser(description='Tune SVM hyperparameters')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. "rami")')
    parser.add_argument('--subsample', type=float, default=0.03, help='Fraction of data to use (default: 0.03)')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset)
    features_dir = config["_resolved"]["features_dir"]
    tuning_dir = config["_resolved"]["tuning_dir"]
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    X, y, groups = load_features_for_tuning(features_dir, subsample_frac=args.subsample)
    
    # Verify coverage
    print(f'Tuning on {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects')
    print(f'Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

    # Use LOSO for subject-independent evaluation
    logo = LeaveOneGroupOut()

    # Test different hyperparameter combinations
    print('\n--- Testing SVM Hyperparameters with LOSO CV ---')
    param_grid = [
        # Best from previous run
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        # Fine-tune around gamma=0.002
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.0015, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.0018, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.0022, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.0025, 'class_weight': 'balanced'},
        # Fine-tune C with best gamma
        {'kernel': 'rbf', 'C': 8.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 12.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 6.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 7.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        # Combinations from 2nd best
        {'kernel': 'rbf', 'C': 5.0, 'gamma': 0.002, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 5.0, 'gamma': 0.0025, 'class_weight': 'balanced'},
        {'kernel': 'rbf', 'C': 7.0, 'gamma': 0.0025, 'class_weight': 'balanced'},
    ]

    results = []
    for params in param_grid:
        start = time.time()
        pipe = make_pipeline(StandardScaler(), SVC(**params, cache_size=500))
        scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, n_jobs=-1)
        elapsed = time.time() - start
        results.append((params, scores.mean(), scores.std(), elapsed))
        print(f'{params}')
        print(f'  LOSO Acc: {scores.mean():.4f} +/- {scores.std():.4f}, Time: {elapsed:.1f}s')

    # Sort by CV accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    print('\n--- Top 5 Results by LOSO Accuracy ---')
    for params, mean_acc, std_acc, elapsed in results[:5]:
        print(f'Acc: {mean_acc:.4f} +/- {std_acc:.4f} | {params}')


if __name__ == '__main__':
    main()
