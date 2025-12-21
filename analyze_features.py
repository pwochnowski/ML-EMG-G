"""Quick feature importance analysis without expensive CV loops.

Usage:
    uv run python analyze_features.py                    # Full analysis
    uv run python analyze_features.py --top 20           # Show top 20 features
    uv run python analyze_features.py --subsample 0.1    # Use 10% of data (faster)
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time


def load_and_pool(results_dir: Path, pattern: str = '*_features_combined.npz'):
    """Load all subject data into one pool."""
    files = sorted(results_dir.glob(pattern))
    Xs, ys = [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        Xs.append(data['X'])
        ys.append(data['y'])
    return np.vstack(Xs), np.concatenate(ys)


def get_feature_names(n_features: int) -> list:
    """Generate feature names (customize based on your feature extraction)."""
    # Based on 105 features = 7 channels x 15 features per channel (typical for combined)
    # Adjust this based on your actual feature set
    n_channels = 7
    features_per_channel = n_features // n_channels
    
    # Common EMG feature names
    base_names = [
        'MAV', 'RMS', 'VAR', 'WL', 'ZC', 'SSC', 'WAMP',  # Time domain
        'MNF', 'MDF', 'PKF',  # Frequency domain
        'AR1', 'AR2', 'AR3', 'AR4',  # Autoregressive
        'IEMG',  # Integrated EMG
    ]
    
    names = []
    for ch in range(n_channels):
        for i in range(features_per_channel):
            feat_name = base_names[i] if i < len(base_names) else f'F{i}'
            names.append(f'Ch{ch+1}_{feat_name}')
    
    # If we have more features than expected, add generic names
    while len(names) < n_features:
        names.append(f'Feature_{len(names)}')
    
    return names[:n_features]


def analyze_features(X, y, top_n: int = 30):
    """Run multiple fast feature importance methods."""
    
    n_features = X.shape[1]
    feature_names = get_feature_names(n_features)
    
    results = {}
    
    # 1. ANOVA F-scores (very fast)
    print("\n1. Computing ANOVA F-scores...")
    t0 = time.time()
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    results['anova_f'] = selector.scores_
    print(f"   Done in {time.time()-t0:.1f}s")
    
    # 2. Mutual Information (moderate speed)
    print("\n2. Computing Mutual Information scores...")
    t0 = time.time()
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    results['mutual_info'] = mi_scores
    print(f"   Done in {time.time()-t0:.1f}s")
    
    # 3. Random Forest importance (fast with subsampling)
    print("\n3. Computing Random Forest importances...")
    t0 = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_scaled, y)
    results['rf_importance'] = rf.feature_importances_
    print(f"   Done in {time.time()-t0:.1f}s")
    
    # Normalize all scores to 0-1 range for comparison
    for key in results:
        scores = results[key]
        results[key] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    
    # Compute aggregate ranking
    aggregate = np.zeros(n_features)
    for scores in results.values():
        # Rank-based aggregation
        ranks = np.argsort(np.argsort(-scores))  # Higher score = lower rank number
        aggregate += ranks
    
    # Lower aggregate = better (ranked higher across methods)
    best_indices = np.argsort(aggregate)
    
    # Print results
    print("\n" + "="*80)
    print(f"TOP {top_n} FEATURES (ranked by consensus across methods)")
    print("="*80)
    print(f"{'Rank':<5} {'Feature':<20} {'ANOVA-F':<12} {'MI':<12} {'RF-Imp':<12} {'Aggregate':<10}")
    print("-"*80)
    
    for rank, idx in enumerate(best_indices[:top_n], 1):
        print(f"{rank:<5} {feature_names[idx]:<20} "
              f"{results['anova_f'][idx]:<12.4f} "
              f"{results['mutual_info'][idx]:<12.4f} "
              f"{results['rf_importance'][idx]:<12.4f} "
              f"{aggregate[idx]:<10.0f}")
    
    # Group analysis by channel
    print("\n" + "="*80)
    print("CHANNEL IMPORTANCE (sum of RF importances)")
    print("="*80)
    
    n_channels = 7
    features_per_channel = n_features // n_channels
    channel_importance = []
    for ch in range(n_channels):
        start = ch * features_per_channel
        end = start + features_per_channel
        ch_imp = np.sum(results['rf_importance'][start:end])
        channel_importance.append((ch + 1, ch_imp))
    
    channel_importance.sort(key=lambda x: -x[1])
    for ch, imp in channel_importance:
        bar = '█' * int(imp * 50 / max(c[1] for c in channel_importance))
        print(f"Channel {ch}: {imp:.4f} {bar}")
    
    # Feature type analysis
    print("\n" + "="*80)
    print("FEATURE TYPE IMPORTANCE (averaged across channels)")
    print("="*80)
    
    feature_type_imp = {}
    base_names = get_feature_names(features_per_channel)
    base_names = [n.split('_')[1] if '_' in n else n for n in base_names]
    
    for feat_idx, name in enumerate(base_names[:features_per_channel]):
        # Get this feature across all channels
        indices = [ch * features_per_channel + feat_idx for ch in range(n_channels)]
        avg_imp = np.mean([results['rf_importance'][i] for i in indices if i < n_features])
        feature_type_imp[name] = avg_imp
    
    sorted_types = sorted(feature_type_imp.items(), key=lambda x: -x[1])
    for name, imp in sorted_types:
        bar = '█' * int(imp * 50 / max(v for v in feature_type_imp.values()))
        print(f"{name:<10}: {imp:.4f} {bar}")
    
    # Save detailed results
    output_file = Path('results/feature_importance.csv')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('feature_idx,feature_name,anova_f,mutual_info,rf_importance,aggregate_rank\n')
        for idx in best_indices:
            f.write(f"{idx},{feature_names[idx]},{results['anova_f'][idx]:.6f},"
                    f"{results['mutual_info'][idx]:.6f},{results['rf_importance'][idx]:.6f},"
                    f"{aggregate[idx]:.0f}\n")
    print(f"\nDetailed results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    recommended_k = [10, 20, 30, 50]
    print(f"Try these --k values with --feat-select kbest: {recommended_k}")
    print(f"\nQuick test commands:")
    for k in recommended_k:
        print(f"  uv run python loso_train.py --models lda --feat-select kbest --k {k} --subsample 0.2")
    
    return best_indices, results


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--results-dir', default='results', help='Directory with feature files')
    parser.add_argument('--top', type=int, default=30, help='Number of top features to show')
    parser.add_argument('--subsample', type=float, help='Fraction to subsample (0-1)')
    args = parser.parse_args()
    
    print("Loading data...")
    X, y = load_and_pool(Path(args.results_dir))
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    if args.subsample:
        n_keep = int(X.shape[0] * args.subsample)
        idx = np.random.choice(X.shape[0], n_keep, replace=False)
        X, y = X[idx], y[idx]
        print(f"Subsampled to: {X.shape[0]} samples")
    
    analyze_features(X, y, top_n=args.top)


if __name__ == '__main__':
    main()
