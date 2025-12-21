"""Master tuning script - runs all model tuning and compares results.

Usage:
    uv run python tune_all.py --dataset rami           # Run all tuners for rami dataset
    uv run python tune_all.py --dataset rami --quick   # Run with smaller parameter grids
"""
import numpy as np
import glob
import argparse
from pathlib import Path
import yaml
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time
import csv

# Try to import optional dependencies
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Skipping XGB tuning.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Skipping LGBM tuning.")


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


def load_data(features_dir: Path, subsample_frac=0.05):
    """Load and subsample data for tuning."""
    data_files = sorted(features_dir.glob('*_features_combined.npz'))
    
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
    
    # Subsample - STRATIFIED by subject AND label
    np.random.seed(42)
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


def tune_model(name, clf_factory, param_grid, X, y, groups, logo, encode_labels=False):
    """Tune a single model type and return best result."""
    print(f'\n{"="*60}')
    print(f'Tuning {name} - {len(param_grid)} configurations')
    print(f'{"="*60}')
    
    y_use = y
    if encode_labels:
        le = LabelEncoder()
        y_use = le.fit_transform(y)
    
    results = []
    for i, params in enumerate(param_grid):
        start = time.time()
        clf = clf_factory(**params)
        pipe = make_pipeline(StandardScaler(), clf)
        
        # Use n_jobs=1 for boosting models (they use internal parallelism)
        n_jobs_cv = 1 if name in ['XGBoost', 'LightGBM'] else -1
        scores = cross_val_score(pipe, X, y_use, cv=logo, groups=groups, n_jobs=n_jobs_cv)
        elapsed = time.time() - start
        results.append((params, scores.mean(), scores.std(), elapsed))
        
        if (i + 1) % max(1, len(param_grid) // 5) == 0:
            print(f'  [{i+1}/{len(param_grid)}] Acc: {scores.mean():.4f} +/- {scores.std():.4f}')
    
    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0]
    print(f'\n  Best {name}: Acc={best[1]:.4f} +/- {best[2]:.4f}')
    print(f'  Params: {best[0]}')
    
    return name, best[0], best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. "rami")')
    parser.add_argument('--quick', action='store_true', help='Use smaller parameter grids')
    args = parser.parse_args()
    
    config = load_dataset_config(args.dataset)
    features_dir = config["_resolved"]["features_dir"]
    tuning_dir = config["_resolved"]["tuning_dir"]
    tuning_dir.mkdir(parents=True, exist_ok=True)
    
    subsample_frac = 0.03 if args.quick else 0.05
    X, y, groups = load_data(features_dir, subsample_frac)
    
    print(f'Tuning on {len(X)} samples, {X.shape[1]} features')
    print(f'{len(np.unique(y))} classes, {len(np.unique(groups))} subjects')
    
    logo = LeaveOneGroupOut()
    all_results = []
    
    # ===== SVM =====
    if args.quick:
        svm_params = [
            {'kernel': 'rbf', 'C': c, 'gamma': g, 'class_weight': 'balanced', 'cache_size': 500}
            for c in [1.0, 10.0] for g in [0.001, 0.002, 0.01]
        ]
    else:
        svm_params = [
            {'kernel': 'rbf', 'C': c, 'gamma': g, 'class_weight': 'balanced', 'cache_size': 500}
            for c in [1.0, 5.0, 10.0, 20.0] for g in [0.001, 0.002, 0.005, 0.01, 'scale']
        ]
    all_results.append(tune_model('SVM', SVC, svm_params, X, y, groups, logo))
    
    # ===== KNN =====
    if args.quick:
        knn_params = [
            {'n_neighbors': k, 'weights': w, 'metric': m, 'n_jobs': -1}
            for k in [5, 11, 21] for w in ['distance'] for m in ['euclidean', 'manhattan']
        ]
    else:
        knn_params = [
            {'n_neighbors': k, 'weights': w, 'metric': m, 'n_jobs': -1}
            for k in [3, 5, 7, 11, 15, 21] for w in ['uniform', 'distance'] 
            for m in ['euclidean', 'manhattan']
        ]
    all_results.append(tune_model('KNN', KNeighborsClassifier, knn_params, X, y, groups, logo))
    
    # ===== Random Forest =====
    if args.quick:
        rf_params = [
            {'n_estimators': n, 'max_depth': d, 'max_features': f, 'n_jobs': -1, 'random_state': 42}
            for n in [200, 500] for d in [20, None] for f in ['sqrt', 0.3]
        ]
    else:
        rf_params = [
            {'n_estimators': n, 'max_depth': d, 'max_features': f, 'n_jobs': -1, 'random_state': 42}
            for n in [100, 200, 300, 500] for d in [10, 20, 30, None] for f in ['sqrt', 'log2', 0.3]
        ]
    all_results.append(tune_model('RandomForest', RandomForestClassifier, rf_params, X, y, groups, logo))
    
    # ===== Extra Trees =====
    if args.quick:
        et_params = [
            {'n_estimators': n, 'max_depth': d, 'max_features': f, 'n_jobs': -1, 'random_state': 42}
            for n in [200, 500] for d in [20, None] for f in ['sqrt', 0.3]
        ]
    else:
        et_params = [
            {'n_estimators': n, 'max_depth': d, 'max_features': f, 'n_jobs': -1, 'random_state': 42}
            for n in [100, 200, 300, 500] for d in [10, 20, 30, None] for f in ['sqrt', 'log2', 0.3]
        ]
    all_results.append(tune_model('ExtraTrees', ExtraTreesClassifier, et_params, X, y, groups, logo))
    
    # ===== XGBoost =====
    if XGBOOST_AVAILABLE:
        if args.quick:
            xgb_params = [
                {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': 0.8,
                 'n_jobs': -1, 'random_state': 42, 'verbosity': 0, 'eval_metric': 'mlogloss'}
                for n in [200, 500] for d in [5, 8] for lr in [0.05, 0.1]
            ]
        else:
            xgb_params = [
                {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': s,
                 'n_jobs': -1, 'random_state': 42, 'verbosity': 0, 'eval_metric': 'mlogloss'}
                for n in [100, 200, 300] for d in [3, 5, 8, 10] for lr in [0.01, 0.05, 0.1]
                for s in [0.8, 1.0]
            ]
        all_results.append(tune_model('XGBoost', XGBClassifier, xgb_params, X, y, groups, logo, encode_labels=True))
    
    # ===== LightGBM =====
    if LIGHTGBM_AVAILABLE:
        if args.quick:
            lgbm_params = [
                {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'num_leaves': nl,
                 'n_jobs': -1, 'random_state': 42, 'verbose': -1}
                for n in [200, 500] for d in [-1, 10] for lr in [0.05, 0.1] for nl in [31, 100]
            ]
        else:
            lgbm_params = [
                {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'num_leaves': nl,
                 'n_jobs': -1, 'random_state': 42, 'verbose': -1}
                for n in [100, 200, 300] for d in [-1, 5, 10] for lr in [0.01, 0.05, 0.1]
                for nl in [31, 50, 100]
            ]
        all_results.append(tune_model('LightGBM', LGBMClassifier, lgbm_params, X, y, groups, logo, encode_labels=True))
    
    # ===== Summary =====
    print('\n' + '='*60)
    print('FINAL SUMMARY - Best Results by Model')
    print('='*60)
    
    all_results.sort(key=lambda x: x[2], reverse=True)
    for name, params, mean_acc, std_acc in all_results:
        print(f'{name:15s}: {mean_acc:.4f} +/- {std_acc:.4f}')
    
    # Save to CSV
    output_file = tuning_dir / 'tuning_summary.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Std', 'Best_Params'])
        for name, params, mean_acc, std_acc in all_results:
            writer.writerow([name, f'{mean_acc:.4f}', f'{std_acc:.4f}', str(params)])
    
    print(f'\nResults saved to {output_file}')
    print('\nBest overall: {} with accuracy {:.4f}'.format(all_results[0][0], all_results[0][2]))


if __name__ == '__main__':
    main()
