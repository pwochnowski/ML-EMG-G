"""Model factory for building classification pipelines.

Provides `build_model(name)` to construct sklearn pipelines for various
classifiers commonly used in EMG classification.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Try to import optional dependencies
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


# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lda": {
        "class": LinearDiscriminantAnalysis,
        "params": {},
    },
    "qda": {
        "class": QuadraticDiscriminantAnalysis,
        "params": {},
    },
    "svm": {
        "class": SVC,
        "params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    },
    "svm-tuned": {
        "class": SVC,
        "params": {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": 0.002,
            "class_weight": "balanced",
            "cache_size": 500,
        },
    },
    "svm-fast": {
        "class": LinearSVC,
        "params": {"C": 1.0, "max_iter": 10000, "dual": "auto"},
    },
    "rf": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "rf-tuned": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 20,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "rf-large": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 30,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "et": {
        "class": ExtraTreesClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "et-tuned": {
        "class": ExtraTreesClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 20,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        },
    },
    "knn": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5, "weights": "distance", "n_jobs": -1},
    },
    "knn-tuned": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": 21,
            "weights": "distance",
            "metric": "manhattan",
            "n_jobs": -1,
        },
    },
    "lr": {
        "class": LogisticRegression,
        "params": {"C": 1.0, "max_iter": 1000, "n_jobs": -1, "random_state": 42},
    },
    "gb": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    },
}

# XGBoost models (if available)
if XGBOOST_AVAILABLE:
    MODEL_CONFIGS["xgb"] = {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        },
    }
    MODEL_CONFIGS["xgb-tuned"] = {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        },
    }

# LightGBM models (if available)
if LIGHTGBM_AVAILABLE:
    MODEL_CONFIGS["lgbm"] = {
        "class": LGBMClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
    }
    MODEL_CONFIGS["lgbm-tuned"] = {
        "class": LGBMClassifier,
        "params": {
            "n_estimators": 500,
            "max_depth": 10,
            "learning_rate": 0.1,
            "num_leaves": 100,
            "subsample": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
    }


def build_model(
    name: str,
    include_scaler: bool = True,
    **override_params,
):
    """Build a classification pipeline.
    
    Args:
        name: Model name (e.g., 'lda', 'svm', 'rf', 'xgb').
        include_scaler: Whether to include StandardScaler in pipeline.
        **override_params: Parameters to override the defaults.
        
    Returns:
        sklearn Pipeline with the requested model.
        
    Raises:
        ValueError: If the model name is unknown.
        ImportError: If the model requires an unavailable package.
    """
    name = name.lower()
    
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list_available_models()}"
        )
    
    config = MODEL_CONFIGS[name]
    
    # Check for optional dependencies
    if name.startswith("xgb") and not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    if name.startswith("lgbm") and not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    # Merge default params with overrides
    params = {**config["params"], **override_params}
    
    # Create classifier
    clf = config["class"](**params)
    
    # Build pipeline
    if include_scaler:
        return make_pipeline(StandardScaler(), clf)
    return clf


def list_available_models() -> List[str]:
    """List all available model names.
    
    Returns:
        Sorted list of model names.
    """
    return sorted(MODEL_CONFIGS.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        name: Model name.
        
    Returns:
        Dictionary with 'class' and 'params' keys.
    """
    name = name.lower()
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_CONFIGS[name].copy()
