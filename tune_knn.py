"""Hyperparameter tuning script for KNN using LOSO cross-validation.

Usage:
    uv run python tune_knn.py --dataset rami
"""
import numpy as np
import argparse
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

from dataset_utils import load_dataset_config, load_features_for_tuning


def main():
    parser = argparse.ArgumentParser(description='Tune KNN hyperparameters')
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

    # Test different hyperparameter combinations
    print('\n--- Testing KNN Hyperparameters with LOSO CV ---')
    param_grid = []

    # Vary n_neighbors
    for k in [3, 5, 7, 9, 11, 15, 21, 31]:
        for weights in ['uniform', 'distance']:
            for metric in ['euclidean', 'manhattan', 'minkowski']:
                if metric == 'minkowski':
                    for p in [1, 2, 3]:  # p=1 is manhattan, p=2 is euclidean
                        param_grid.append({
                            'n_neighbors': k,
                            'weights': weights,
                            'metric': 'minkowski',
                            'p': p
                        })
                else:
                    param_grid.append({
                        'n_neighbors': k,
                        'weights': weights,
                        'metric': metric
                    })

    # Remove duplicates (manhattan == minkowski p=1, euclidean == minkowski p=2)
    seen = set()
    unique_params = []
    for p in param_grid:
        key = (p['n_neighbors'], p['weights'], p['metric'], p.get('p', 2))
        if key not in seen:
            seen.add(key)
            unique_params.append(p)
    param_grid = unique_params

    print(f'Testing {len(param_grid)} parameter combinations...\n')

    results = []
    for i, params in enumerate(param_grid):
        start = time.time()
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(**params, n_jobs=-1))
        scores = cross_val_score(pipe, X, y, cv=logo, groups=groups, n_jobs=-1)
        elapsed = time.time() - start
        results.append((params, scores.mean(), scores.std(), elapsed))
        
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
    print(f"clf = KNeighborsClassifier(n_neighbors={best['n_neighbors']}, weights='{best['weights']}', metric='{best['metric']}'" + 
          (f", p={best['p']}" if 'p' in best else "") + ", n_jobs=-1)")


if __name__ == '__main__':
    main()
