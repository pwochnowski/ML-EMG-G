"""CLI for training and evaluating EMG classification models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

from ..config import load_config
from ..evaluation import run_loso_cv, run_within_subject_cv, load_and_pool_subjects
from ..evaluation.cross_validation import save_results_csv
from ..models import list_available_models


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main(args: Optional[List[str]] = None):
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate EMG classification models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing feature files"
    )
    parser.add_argument(
        "--pattern",
        default="*_features_combined.npz",
        help="Glob pattern for feature files"
    )
    parser.add_argument(
        "--subjects",
        help="Comma-separated list of subjects to include"
    )
    parser.add_argument(
        "--models",
        help=f"Comma-separated model names. Available: {', '.join(list_available_models())}"
    )
    parser.add_argument(
        "--cv-type",
        choices=["loso", "within_subject", "none"],
        default="loso",
        help="Cross-validation strategy"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for within-subject CV"
    )
    parser.add_argument(
        "--normalizer",
        choices=["standard", "percentile", "channel", "adaptive"],
        default="standard",
        help="Normalization strategy"
    )
    parser.add_argument(
        "--feat-select",
        choices=["none", "kbest", "pca"],
        default="none",
        help="Feature selection method"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of features for SelectKBest"
    )
    parser.add_argument(
        "--pca-n",
        type=int,
        help="Number of PCA components"
    )
    parser.add_argument(
        "--calibration",
        type=int,
        default=0,
        help="Calibration samples per class from test subject"
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--save-models",
        type=Path,
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parsed = parser.parse_args(args)
    setup_logging(parsed.verbose)
    logger = logging.getLogger(__name__)
    
    # Load config if provided
    if parsed.config:
        config = load_config(parsed.config)
        data_dir = config.results_dir
        model_names = [m.name for m in config.models]
        normalizer = config.normalizer.type
        cv_type = config.evaluation.cv_type
    else:
        data_dir = parsed.data_dir
        model_names = parsed.models.split(",") if parsed.models else ["lda"]
        normalizer = parsed.normalizer
        cv_type = parsed.cv_type
    
    # Parse subjects
    subjects = parsed.subjects.split(",") if parsed.subjects else None
    
    # Load data
    logger.info(f"Loading features from {data_dir}")
    X, y, groups, positions = load_and_pool_subjects(
        results_dir=data_dir,
        pattern=parsed.pattern,
        subjects=subjects,
    )
    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Subjects: {list(set(groups))}")
    logger.info(f"Classes: {list(set(y))}")
    
    # Run cross-validation
    if cv_type == "loso":
        results = run_loso_cv(
            X=X,
            y=y,
            groups=groups,
            model_names=model_names,
            normalizer=normalizer,
            feature_selection=parsed.feat_select,
            k_best=parsed.k,
            pca_components=parsed.pca_n,
            calibration_samples=parsed.calibration,
            save_models_dir=parsed.save_models,
        )
    elif cv_type == "within_subject":
        results = run_within_subject_cv(
            X=X,
            y=y,
            groups=groups,
            model_names=model_names,
            n_folds=parsed.n_folds,
            normalizer=normalizer,
            feature_selection=parsed.feat_select,
            k_best=parsed.k,
            pca_components=parsed.pca_n,
        )
    else:
        # No CV - just report
        logger.info("No cross-validation requested")
        results = {}
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for model_name, data in results.items():
        if cv_type == "loso":
            print(
                f"{model_name}: {data['mean_accuracy']:.4f} "
                f"± {data['std_accuracy']:.4f}"
            )
        else:
            print(
                f"{model_name}: {data['overall_mean']:.4f} "
                f"± {data['overall_std']:.4f} (across subjects)"
            )
    
    # Save results
    if parsed.out_csv:
        save_results_csv(results, parsed.out_csv, cv_type)


if __name__ == "__main__":
    main()
