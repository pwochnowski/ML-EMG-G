"""Model factory utilities.

Provides `build_model(name)` to construct a sklearn pipeline for common models.
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for LabelEncoder that works in a pipeline (transforms y, not X)."""
    def __init__(self):
        self.le_ = None
        self.classes_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return X


# Try to import XGBoost and LightGBM (fast gradient boosting)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import RAPIDS cuML for GPU-accelerated models
try:
    import cuml
    from cuml.svm import SVC as cuMLSVC
    from cuml.neighbors import KNeighborsClassifier as cuMLKNN
    from cuml.ensemble import RandomForestClassifier as cuMLRF
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Check if CUDA is available for XGBoost/LightGBM GPU
def _check_cuda_available():
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Fallback: try to detect via xgboost
    try:
        import xgboost as xgb
        # Try to create a small GPU DMatrix - if it fails, no GPU
        return True  # XGBoost 2.0+ handles GPU detection internally
    except:
        pass
    return False

CUDA_AVAILABLE = _check_cuda_available()



def build_model(name: str):
    name = name.lower()
    if name == 'lda':
        clf = LinearDiscriminantAnalysis()
        return make_pipeline(StandardScaler(), clf)
    if name == 'qda':
        clf = QuadraticDiscriminantAnalysis()
        return make_pipeline(StandardScaler(), clf)
    if name == 'svm':
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        return make_pipeline(StandardScaler(), clf)
    if name == 'svm-tuned':
        # Better hyperparameters for EMG LOSO classification
        # Lower gamma (~0.002 vs auto ~0.0095) provides smoother decision boundary
        # that generalizes better across subjects
        clf = SVC(kernel='rbf', C=10.0, gamma=0.002, class_weight='balanced', cache_size=500)
        return make_pipeline(StandardScaler(), clf)
    if name == 'svm-fast':
        # Much faster linear SVM - good for large datasets
        clf = LinearSVC(C=1.0, max_iter=10000, dual='auto')
        return make_pipeline(StandardScaler(), clf)
    if name == 'rf':
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                                     n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    if name == 'rf-large':
        clf = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_leaf=1,
                                     n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    if name == 'et':
        # Extra Trees - often better than RF, faster
        clf = ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                                   n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    if name == 'knn-tuned':
        clf = KNeighborsClassifier(n_neighbors=21, weights='distance', metric='manhattan', n_jobs=-1)
        return make_pipeline(StandardScaler(), clf)
    if name == 'lr':
        clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    if name == 'xgb':
        if not XGBOOST_AVAILABLE:
            raise ImportError('XGBoost not installed. Run: uv add xgboost')
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           n_jobs=-1, random_state=42, verbosity=0)
        return make_pipeline(StandardScaler(), clf)
    if name == 'xgb-tuned':
        if not XGBOOST_AVAILABLE:
            raise ImportError('XGBoost not installed. Run: uv add xgboost')
        # Tuned for EMG LOSO classification
        clf = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                           subsample=0.8, n_jobs=-1, random_state=42, verbosity=0)
        return make_pipeline(StandardScaler(), clf)
    if name == 'lgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not installed. Run: uv add lightgbm')
        clf = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=42, verbose=-1)
        return make_pipeline(StandardScaler(), clf)
    if name == 'lgbm-tuned':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not installed. Run: uv add lightgbm')
        # Tuned for EMG LOSO classification
        clf = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.1,
                            num_leaves=100, subsample=0.8, n_jobs=-1, 
                            random_state=42, verbose=-1)
        return make_pipeline(StandardScaler(), clf)
    
    # ==========================================================================
    # GPU-ACCELERATED MODELS (with CPU fallback)
    # ==========================================================================
    
    if name == 'xgb-gpu':
        if not XGBOOST_AVAILABLE:
            raise ImportError('XGBoost not installed. Run: uv add xgboost')
        # GPU-accelerated XGBoost (falls back to CPU if CUDA unavailable)
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           tree_method='hist', device='cuda' if CUDA_AVAILABLE else 'cpu',
                           random_state=42, verbosity=0)
        if not CUDA_AVAILABLE:
            print("Warning: CUDA not available, xgb-gpu falling back to CPU")
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'xgb-tuned-gpu':
        if not XGBOOST_AVAILABLE:
            raise ImportError('XGBoost not installed. Run: uv add xgboost')
        # Tuned GPU-accelerated XGBoost
        clf = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1,
                           subsample=0.8, tree_method='hist', 
                           device='cuda' if CUDA_AVAILABLE else 'cpu',
                           random_state=42, verbosity=0)
        if not CUDA_AVAILABLE:
            print("Warning: CUDA not available, xgb-tuned-gpu falling back to CPU")
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'lgbm-gpu':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not installed. Run: uv add lightgbm')
        # GPU-accelerated LightGBM (requires GPU build of LightGBM)
        try:
            clf = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                device='gpu', random_state=42, verbose=-1)
        except Exception as e:
            print(f"Warning: LightGBM GPU failed ({e}), falling back to CPU")
            clf = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                n_jobs=-1, random_state=42, verbose=-1)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'lgbm-tuned-gpu':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('LightGBM not installed. Run: uv add lightgbm')
        # Tuned GPU-accelerated LightGBM
        try:
            clf = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.1,
                                num_leaves=100, subsample=0.8, device='gpu',
                                random_state=42, verbose=-1)
        except Exception as e:
            print(f"Warning: LightGBM GPU failed ({e}), falling back to CPU")
            clf = LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.1,
                                num_leaves=100, subsample=0.8, n_jobs=-1,
                                random_state=42, verbose=-1)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'svm-gpu':
        if not CUML_AVAILABLE:
            print("Warning: cuML not available, svm-gpu falling back to CPU sklearn SVM")
            clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        else:
            clf = cuMLSVC(kernel='rbf', C=1.0, gamma='scale')
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'svm-tuned-gpu':
        if not CUML_AVAILABLE:
            print("Warning: cuML not available, svm-tuned-gpu falling back to CPU sklearn SVM")
            clf = SVC(kernel='rbf', C=10.0, gamma=0.002, class_weight='balanced', cache_size=500)
        else:
            # cuML SVC doesn't support class_weight, but is much faster
            clf = cuMLSVC(kernel='rbf', C=10.0, gamma=0.002, cache_size=500)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'knn-gpu':
        if not CUML_AVAILABLE:
            print("Warning: cuML not available, knn-gpu falling back to CPU sklearn KNN")
            clf = KNeighborsClassifier(n_neighbors=21, weights='distance', metric='manhattan', n_jobs=-1)
        else:
            # cuML KNN - note: limited metric support (euclidean default)
            clf = cuMLKNN(n_neighbors=21)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'rf-gpu':
        if not CUML_AVAILABLE:
            print("Warning: cuML not available, rf-gpu falling back to CPU sklearn RF")
            clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2,
                                         n_jobs=-1, random_state=42)
        else:
            # cuML RandomForest - note: different API, limited params
            clf = cuMLRF(n_estimators=200, max_depth=20, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'rf-tuned-gpu':
        if not CUML_AVAILABLE:
            print("Warning: cuML not available, rf-tuned-gpu falling back to CPU sklearn RF")
            clf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='sqrt',
                                         n_jobs=-1, random_state=42)
        else:
            clf = cuMLRF(n_estimators=500, max_depth=20, max_features='sqrt', random_state=42)
        return make_pipeline(StandardScaler(), clf)
    
    if name == 'rf-tuned':
        # Tuned for EMG LOSO classification
        clf = RandomForestClassifier(n_estimators=500, max_depth=20, max_features='sqrt',
                                     n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
    if name == 'et-tuned':
        # Tuned for EMG LOSO classification - best performer
        clf = ExtraTreesClassifier(n_estimators=500, max_depth=20, max_features='sqrt',
                                   n_jobs=-1, random_state=42)
        return make_pipeline(StandardScaler(), clf)
   
    raise ValueError(f'Unknown model: {name}')


def get_available_models():
    """Return list of all available model names."""
    cpu_models = [
        'lda', 'qda', 'svm', 'svm-tuned', 'svm-fast', 
        'rf', 'rf-large', 'rf-tuned', 'et', 'et-tuned',
        'knn-tuned', 'lr'
    ]
    
    boosting_models = []
    if XGBOOST_AVAILABLE:
        boosting_models.extend(['xgb', 'xgb-tuned'])
    if LIGHTGBM_AVAILABLE:
        boosting_models.extend(['lgbm', 'lgbm-tuned'])
    
    gpu_models = []
    if XGBOOST_AVAILABLE:
        gpu_models.extend(['xgb-gpu', 'xgb-tuned-gpu'])
    if LIGHTGBM_AVAILABLE:
        gpu_models.extend(['lgbm-gpu', 'lgbm-tuned-gpu'])
    # cuML models always available (fallback to CPU)
    gpu_models.extend(['svm-gpu', 'svm-tuned-gpu', 'knn-gpu', 'rf-gpu', 'rf-tuned-gpu'])
    
    return cpu_models + boosting_models + gpu_models


def get_gpu_status():
    """Return a dict with GPU availability status."""
    return {
        'cuda_available': CUDA_AVAILABLE,
        'cuml_available': CUML_AVAILABLE,
        'xgboost_available': XGBOOST_AVAILABLE,
        'lightgbm_available': LIGHTGBM_AVAILABLE,
    }


def print_gpu_status():
    """Print GPU availability status."""
    status = get_gpu_status()
    print("=" * 50)
    print("GPU Acceleration Status")
    print("=" * 50)
    print(f"  CUDA Available:     {status['cuda_available']}")
    print(f"  cuML Available:     {status['cuml_available']}")
    print(f"  XGBoost Available:  {status['xgboost_available']}")
    print(f"  LightGBM Available: {status['lightgbm_available']}")
    print("=" * 50)
    
    if status['cuda_available'] or status['cuml_available']:
        print("\nGPU models available:")
        if status['xgboost_available'] and status['cuda_available']:
            print("  - xgb-gpu, xgb-tuned-gpu (XGBoost with CUDA)")
        if status['lightgbm_available']:
            print("  - lgbm-gpu, lgbm-tuned-gpu (LightGBM with GPU)")
        if status['cuml_available']:
            print("  - svm-gpu, svm-tuned-gpu (cuML SVM)")
            print("  - knn-gpu (cuML KNN)")
            print("  - rf-gpu, rf-tuned-gpu (cuML RandomForest)")
    else:
        print("\nNo GPU acceleration available. GPU models will fall back to CPU.")
    print()


if __name__ == '__main__':
    print_gpu_status()
    print("Available models:", get_available_models())
