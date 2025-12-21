"""Hyperparameter tuning script for Random Forest using LOSO cross-validation.

Usage:
    uv run python tune_rf.py --dataset rami
"""
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

from dataset_utils import load_dataset_config, load_features_for_tuning


def main():
    parser = argparse.ArgumentParser(description='Tune Random Forest and Extra Trees hyperparameters')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. "rami")')
    parser.add_argument('--subsample', type=float, default=0.05, help='Fraction of data to use (default: 0.05)')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset)
    features_dir = config["_resolved"]["features_dir"]
    tuning_dir = config["_resolved"]["tuning_dir"]
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    X, y, groups = load_features_for_tuning(features_dir, subsample_frac=args.subsample)

    print(f'Tuning on {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes, {len(np.unique(groups))} subjects')
    print(f'Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

    # Use LOSO for subject-independent evaluation
    logo = LeaveOneGroupOut()

    # Test different hyperparameter combinations for both RF and ET
    print('\n--- Testing Random Forest & Extra Trees Hyperparameters with LOSO CV ---')

    # Parameter combinations to try
    n_estimators_list = [100, 200, 300, 500]
    max_depth_list = [10, 20, 30, None]
    min_samples_split_list = [2, 5, 10]
    min_samples_leaf_list = [1, 2, 4]
    max_features_list = ['sqrt', 'log2', 0.3]

    # Generate a reasonable subset of combinations (full grid would be too large)
    param_grid = []

    # Core combinations for Random Forest
    for n_est in n_estimators_list:
        for max_depth in max_depth_list:
            for max_feat in max_features_list:
                param_grid.append({
                    'model': 'rf',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': max_feat
                })

    # Add some variations with different min_samples
    for min_split in min_samples_split_list:
        for min_leaf in min_samples_leaf_list:
            param_grid.append({
                'model': 'rf',
                'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'max_features': 'sqrt'
        })

# Extra Trees variations (often better than RF)
    for n_est in [200, 300, 500]:
        for max_depth in [20, 30, None]:
            for max_feat in ['sqrt', 'log2', 0.3]:
                param_grid.append({
                    'model': 'et',
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': max_feat
                })

    # Remove duplicates
    seen = set()
    unique_params = []
    for p in param_grid:
        key = (p['model'], p['n_estimators'], p['max_depth'], 
               p['min_samples_split'], p['min_samples_leaf'], p['max_features'])
        if key not in seen:
            seen.add(key)
            unique_params.append(p)
    param_grid = unique_params

    print(f'Testing {len(param_grid)} parameter combinations...\n')

    results = []
    for i, params in enumerate(param_grid):
        start = time.time()
        
        model_type = params.pop('model')
        if model_type == 'rf':
            clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        else:
            clf = ExtraTreesClassifier(**params, n_jobs=-1, random_state=42)
        params['model'] = model_type  # Put it back for results
        
        pipe = make_pipeline(StandardScaler(), clf)
        scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, n_jobs=-1)
        elapsed = time.time() - start
        results.append((params.copy(), scores.mean(), scores.std(), elapsed))
        
        if (i + 1) % 10 == 0 or i == 0:
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
    model_class = 'RandomForestClassifier' if best['model'] == 'rf' else 'ExtraTreesClassifier'
    print(f"clf = {model_class}(n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, " +
          f"min_samples_split={best['min_samples_split']}, min_samples_leaf={best['min_samples_leaf']}, " +
          f"max_features={repr(best['max_features'])}, n_jobs=-1, random_state=42)")

    # Also show best RF and best ET separately
    print('\n--- Best Random Forest ---')
    rf_results = [r for r in results if r[0]['model'] == 'rf']
    if rf_results:
        best_rf = rf_results[0][0]
        print(f"Acc: {rf_results[0][1]:.4f} | clf = RandomForestClassifier(n_estimators={best_rf['n_estimators']}, " +
              f"max_depth={best_rf['max_depth']}, max_features={repr(best_rf['max_features'])}, n_jobs=-1, random_state=42)")

    print('\n--- Best Extra Trees ---')
    et_results = [r for r in results if r[0]['model'] == 'et']
    if et_results:
        best_et = et_results[0][0]
        print(f"Acc: {et_results[0][1]:.4f} | clf = ExtraTreesClassifier(n_estimators={best_et['n_estimators']}, " +
              f"max_depth={best_et['max_depth']}, max_features={repr(best_et['max_features'])}, n_jobs=-1, random_state=42)")


if __name__ == '__main__':
    main()