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
    if name == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
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
