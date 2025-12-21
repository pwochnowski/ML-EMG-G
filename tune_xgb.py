"""Hyperparameter tuning script for XGBoost using LOSO cross-validation.

Usage:
    uv run python tune_xgb.py --dataset rami
"""
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Run: uv add xgboost")
    exit(1)

from dataset_utils import load_dataset_config, load_features_for_tuning


def main():
    parser = argparse.ArgumentParser(description='Tune XGBoost hyperparameters')
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
    print('\n--- Testing XGBoost Hyperparameters with LOSO CV ---')

    # Parameter grid
    n_estimators_list = [100, 200, 300, 500]
    max_depth_list = [3, 5, 6, 8, 10]
    learning_rate_list = [0.01, 0.05, 0.1, 0.2]
    subsample_list = [0.7, 0.8, 0.9, 1.0]
    colsample_bytree_list = [0.7, 0.8, 0.9, 1.0]
    reg_alpha_list = [0, 0.1, 1.0]  # L1 regularization
    reg_lambda_list = [1.0, 2.0, 5.0]  # L2 regularization

    # Generate a reasonable subset of combinations
    param_grid = []

    # Core combinations
    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            for lr in learning_rate_list:
                param_grid.append({
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'learning_rate': lr,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1.0
                })

    # Add regularization variations on best typical settings
    for reg_alpha in reg_alpha_list:
        for reg_lambda in reg_lambda_list:
            param_grid.append({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            })

    # Add subsample/colsample variations
    for subsample in subsample_list:
        for colsample in colsample_bytree_list:
            param_grid.append({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': subsample,
                'colsample_bytree': colsample,
                'reg_alpha': 0,
                'reg_lambda': 1.0
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
        
        clf = XGBClassifier(**params, n_jobs=-1, random_state=42, verbosity=0,
                            use_label_encoder=False, eval_metric='mlogloss')
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, n_jobs=1)  # XGB uses internal parallelism
        elapsed = time.time() - start
        results.append((params.copy(), scores.mean(), scores.std(), elapsed))
        
        if (i + 1) % 20 == 0 or i == 0:
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
    print(f"clf = XGBClassifier(n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, " +
          f"learning_rate={best['learning_rate']}, subsample={best['subsample']}, " +
          f"colsample_bytree={best['colsample_bytree']}, reg_alpha={best['reg_alpha']}, " +
          f"reg_lambda={best['reg_lambda']}, n_jobs=-1, random_state=42, verbosity=0)")


if __name__ == '__main__':
    main()
