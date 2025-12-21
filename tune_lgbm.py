"""Hyperparameter tuning script for LightGBM using LOSO cross-validation.

Usage:
    uv run python tune_lgbm.py --dataset rami
"""
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Run: uv add lightgbm")
    exit(1)

from dataset_utils import load_dataset_config, load_features_for_tuning


def main():
    parser = argparse.ArgumentParser(description='Tune LightGBM hyperparameters')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. "rami")')
    parser.add_argument('--subsample', type=float, default=0.05, help='Fraction of data to use (default: 0.05)')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset)
    features_dir = config["_resolved"]["features_dir"]
    tuning_dir = config["_resolved"]["tuning_dir"]
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    X, y, groups = load_features_for_tuning(features_dir, subsample_frac=args.subsample, encode_labels=True)

    print(f'Tuning on {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects')
    print(f'Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

    # Use LOSO for subject-independent evaluation
    logo = LeaveOneGroupOut()

    # Test different hyperparameter combinations
    print('\n--- Testing LightGBM Hyperparameters with LOSO CV ---')

    # Parameter grid - LightGBM specific
    n_estimators_list = [100, 200, 300, 500]
    max_depth_list = [-1, 5, 8, 10, 15]  # -1 means no limit
    learning_rate_list = [0.01, 0.05, 0.1, 0.2]
    num_leaves_list = [31, 50, 100, 150]  # LightGBM specific - controls tree complexity
    subsample_list = [0.7, 0.8, 0.9, 1.0]  # called bagging_fraction in lgbm
    colsample_bytree_list = [0.7, 0.8, 0.9, 1.0]  # called feature_fraction in lgbm
    reg_alpha_list = [0, 0.1, 1.0]  # L1 regularization
    reg_lambda_list = [0, 1.0, 5.0]  # L2 regularization
    min_child_samples_list = [5, 10, 20]  # min data in leaf

    # Generate a reasonable subset of combinations
    param_grid = []

    # Core combinations: n_estimators, max_depth, learning_rate, num_leaves
    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            for lr in learning_rate_list:
                for num_leaves in num_leaves_list:
                    param_grid.append({
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': lr,
                    'num_leaves': num_leaves,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 0,
                    'min_child_samples': 10
                })

    # Regularization variations
    for reg_alpha in reg_alpha_list:
        for reg_lambda in reg_lambda_list:
            param_grid.append({
                'n_estimators': 200,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'min_child_samples': 10
            })

    # Subsample/colsample variations
    for subsample in subsample_list:
        for colsample in colsample_bytree_list:
            param_grid.append({
                'n_estimators': 200,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': subsample,
                'colsample_bytree': colsample,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'min_child_samples': 10
            })

    # Min child samples variations
    for min_child in min_child_samples_list:
        param_grid.append({
            'n_estimators': 200,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'min_child_samples': min_child
        })

    # Remove duplicates
    seen = set()
    unique_params = []
    for p in param_grid:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            unique_params.append(p)
    param_grid = unique_params

    print(f'Testing {len(param_grid)} parameter combinations...\n')

    results = []
    for i, params in enumerate(param_grid):
        start = time.time()
        
        clf = LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, n_jobs=1)  # LGBM uses internal parallelism
        elapsed = time.time() - start
        results.append((params.copy(), scores.mean(), scores.std(), elapsed))
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f'[{i+1}/{len(param_grid)}] {params}')
            print(f'  LOSO Acc: {scores.mean():.4f} +/- {scores.std():.4f}, Time: {elapsed:.1f}s')

    # Sort by CV accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    print('\n--- Top 10 Results by LOSO Accuracy ---')
    for params, mean_acc, std_acc, elapsed in results[:10]:
        print(f'Acc: {mean_acc:.4f} +/- {std_acc:.4f} | {params}')

    # Print best parameters for models.py
    print('\n--- Best Parameters for models.py ---')
    best = results[0][0]
    print(f"clf = LGBMClassifier(n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, " +
          f"learning_rate={best['learning_rate']}, num_leaves={best['num_leaves']}, " +
          f"subsample={best['subsample']}, colsample_bytree={best['colsample_bytree']}, " +
          f"reg_alpha={best['reg_alpha']}, reg_lambda={best['reg_lambda']}, " +
          f"min_child_samples={best['min_child_samples']}, n_jobs=-1, random_state=42, verbose=-1)")


if __name__ == '__main__':
    main()
