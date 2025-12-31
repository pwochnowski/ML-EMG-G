#!/usr/bin/env python3
"""Evaluate different preprocessing parameter configurations.

This script systematically tests different combinations of preprocessing
parameters (bandpass filter, notch filter) and evaluates their impact on
classification performance using LOSO cross-validation.

Usage:
    python evaluate_preprocessing.py --dataset rami --model lda
    python evaluate_preprocessing.py --dataset rami --model svm-gpu --quick
    python evaluate_preprocessing.py --dataset rami --model lda --parallel 4
    
The script will:
1. Extract features with each preprocessing configuration (in parallel batches)
2. Run LOSO evaluation
3. Save results to a summary CSV
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import yaml
import numpy as np
from datetime import datetime
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Lock for thread-safe printing
print_lock = threading.Lock()


@dataclass
class PreprocessingParams:
    """Configuration for a preprocessing experiment."""
    name: str  # Descriptive name
    bandpass_enabled: bool = True
    bandpass_lowcut: float = 20.0
    bandpass_highcut: float = 500.0
    bandpass_order: int = 4
    notch_enabled: bool = True
    notch_freq: float = 50.0
    notch_q: float = 30.0


import subprocess
import sys

def run_command(command, env):
    # Use Popen for real-time interaction
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout stream
        text=True,                 # Decode output as text (Python 3.6+)
        bufsize=1                  # Line buffer output
    )

    outbuf=[]
    # Read output line by line as it is produced
    for line in iter(process.stdout.readline, ''):
        outbuf.append(line)
        print(line.rstrip(), flush=True) # Print line immediately

    # Wait for the process to finish and get the return code
    process.stdout.close()
    return_code = process.wait()
    
    return return_code, outbuf


# Define preprocessing configurations to test
# Based on EMG signal processing literature
PREPROCESSING_CONFIGS = [
    # Baseline: No preprocessing
    PreprocessingParams(
        name="no_preprocessing",
        bandpass_enabled=False,
        notch_enabled=False,
    ),
    
    # Standard EMG preprocessing (default)
    PreprocessingParams(
        name="standard_20_500",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # More aggressive low-cut (removes more motion artifacts)
    PreprocessingParams(
        name="highpass_30_500",
        bandpass_enabled=True,
        bandpass_lowcut=30.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Even more aggressive (some studies use 50Hz low-cut)
    PreprocessingParams(
        name="highpass_50_500",
        bandpass_enabled=True,
        bandpass_lowcut=50.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Conservative low-cut (preserves more signal)
    PreprocessingParams(
        name="highpass_10_500",
        bandpass_enabled=True,
        bandpass_lowcut=10.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Lower high-cut (might reduce noise)
    PreprocessingParams(
        name="bandpass_20_450",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=450.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Narrower band (common in some EMG studies)
    PreprocessingParams(
        name="bandpass_20_400",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=400.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Higher filter order (sharper cutoff)
    PreprocessingParams(
        name="order6_20_500",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=6,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Lower filter order (gentler cutoff)
    PreprocessingParams(
        name="order2_20_500",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=2,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Bandpass only, no notch
    PreprocessingParams(
        name="bandpass_only_20_500",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=False,
    ),
    
    # Notch only, no bandpass
    PreprocessingParams(
        name="notch_only",
        bandpass_enabled=False,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=30.0,
    ),
    
    # Wider notch (lower Q = more bandwidth removed)
    PreprocessingParams(
        name="wide_notch_q15",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=15.0,
    ),
    
    # Narrower notch (higher Q = less bandwidth removed)
    PreprocessingParams(
        name="narrow_notch_q50",
        bandpass_enabled=True,
        bandpass_lowcut=20.0,
        bandpass_highcut=500.0,
        bandpass_order=4,
        notch_enabled=True,
        notch_freq=50.0,
        notch_q=50.0,
    ),
]

# Quick test configs (subset for faster evaluation)
QUICK_CONFIGS = [
    PreprocessingParams(name="no_preprocessing", bandpass_enabled=False, notch_enabled=False),
    PreprocessingParams(name="standard_20_500", bandpass_enabled=True, bandpass_lowcut=20.0, 
                        bandpass_highcut=500.0, bandpass_order=4, notch_enabled=True, 
                        notch_freq=50.0, notch_q=30.0),
    PreprocessingParams(name="highpass_30_500", bandpass_enabled=True, bandpass_lowcut=30.0, 
                        bandpass_highcut=500.0, bandpass_order=4, notch_enabled=True, 
                        notch_freq=50.0, notch_q=30.0),
    # PreprocessingParams(name="bandpass_only_20_500", bandpass_enabled=True, bandpass_lowcut=20.0, 
    #                     bandpass_highcut=500.0, bandpass_order=4, notch_enabled=False),
]


def update_features_yaml(params: PreprocessingParams, features_yaml_path: Path) -> None:
    """Update the preprocessing section of features.yaml with new parameters."""
    with open(features_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update preprocessing config
    config['preprocessing'] = {
        'bandpass': {
            'enabled': params.bandpass_enabled,
            'lowcut': params.bandpass_lowcut,
            'highcut': params.bandpass_highcut,
            'order': params.bandpass_order,
        },
        'notch': {
            'enabled': params.notch_enabled,
            'freq': params.notch_freq,
            'q_factor': params.notch_q,
        }
    }
    
    with open(features_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_temp_features_yaml(params: PreprocessingParams, base_features_yaml: Path) -> Path:
    """Create a temporary features.yaml with custom preprocessing parameters.
    
    Returns the path to the temporary file. Caller is responsible for cleanup.
    """
    # Read base config
    with open(base_features_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update preprocessing config
    config['preprocessing'] = {
        'bandpass': {
            'enabled': params.bandpass_enabled,
            'lowcut': params.bandpass_lowcut,
            'highcut': params.bandpass_highcut,
            'order': params.bandpass_order,
        },
        'notch': {
            'enabled': params.notch_enabled,
            'freq': params.notch_freq,
            'q_factor': params.notch_q,
        }
    }
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix=f'preproc_{params.name}_')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return Path(temp_path)


def extract_features(dataset: str, output_subdir: str, config_path: Path, 
                     disable_gpu: bool = False, preprocess_config_path: Optional[Path] = None,
                     quiet: bool = False) -> bool:
    """Run feature extraction with the current preprocessing config.
    
    Args:
        dataset: Dataset name
        output_subdir: Subdirectory for output features
        config_path: Path to dataset config.yaml
        disable_gpu: Whether to disable GPU
        preprocess_config_path: Optional path to custom features.yaml for preprocessing
        quiet: If True, suppress output
    """
    cmd = [
        "uv", "run", "python", "-m", "src.emg_classification.cli.extract",
        "--config", str(config_path),
        "--feature-set", "experimental",
        "--preprocess",
        "--out-dir", f"datasets/{dataset}/features",
        "--subset-name", output_subdir,
    ]
    
    # Add custom preprocess config if provided
    if preprocess_config_path:
        cmd.extend(["--preprocess-config", str(preprocess_config_path)])
    
    env = os.environ.copy()
    if disable_gpu:
        env["EMG_DISABLE_GPU"] = "1"

    if not quiet:
        with print_lock:
            logger.info(f"Extracting features to {output_subdir}...")

    # print(' '.join(cmd)) 

    if quiet:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return result.returncode == 0
    else:
        returncode, _ = run_command(cmd, env=env)
        return returncode == 0


def run_loso_evaluation(dataset: str, model: str, feature_subdir: str,
                        subjects: Optional[List[str]] = None,
                        quiet: bool = False) -> dict:
    """Run LOSO evaluation and return results."""
    cmd = [
        "uv", "run", "python", "loso_train.py",
        "--dataset", dataset,
        "--model", model,
        "--subsample", "0.75",
        "--feature-subdir", f"default/{feature_subdir}",
        "--feature-set",  f"{feature_subdir}"
    ]
    
    if subjects:
        cmd.extend(["--subjects", ",".join(subjects)])

    if not quiet:
        _, cmd_output = run_command(cmd, env=os.environ.copy())
        with print_lock:
            logger.info(f"Running LOSO with {model} on {feature_subdir}...")
    else: 
        _, cmd_output = run_command(cmd, env=os.environ.copy())
        with print_lock:
            logger.info(f"Running LOSO with {model} on {feature_subdir}...")
        # result = subprocess.run(cmd, capture_output=True, text=True)
        # cmd_output = result.stdout
    
    # Parse results from output
    results = {
        'accuracy': None,
        'std': None,
        'f1': None,
        'n_folds': None,
    }
    
    for line in cmd_output.split('\n'):
        # Look for summary line like: "Model: lda  LOSO accuracy: 0.6269 ± 0.1564  macro-F1: 0.6063  (folds=11)"
        if f"Model: {model}" in line and "LOSO accuracy:" in line:
            try:
                # Parse accuracy
                acc_part = line.split("LOSO accuracy:")[1].split("±")[0].strip()
                results['accuracy'] = float(acc_part)
                
                # Parse std if present
                if "±" in line:
                    std_part = line.split("±")[1].split("macro-F1")[0].strip()
                    results['std'] = float(std_part)
                
                # Parse F1 if present
                if "macro-F1:" in line:
                    f1_part = line.split("macro-F1:")[1].split("(")[0].strip()
                    results['f1'] = float(f1_part)
                
                # Parse n_folds if present
                if "(folds=" in line:
                    folds_part = line.split("(folds=")[1].split(")")[0].strip()
                    results['n_folds'] = int(folds_part)
            except (IndexError, ValueError) as e:
                if not quiet:
                    logger.warning(f"Could not parse results from: {line}")
    
    if result.returncode != 0:
        if not quiet:
            logger.warning(f"LOSO evaluation returned non-zero exit code")
            logger.debug(f"stderr: {result.stderr}")
    
    return results


def process_single_config(
    params: PreprocessingParams,
    dataset: str,
    model: str,
    config_path: Path,
    features_yaml_path: Path,
    disable_gpu: bool = False,
    keep_features: bool = False,
    quiet: bool = True,
) -> dict:
    """Process a single preprocessing configuration (for parallel execution).
    
    Creates a temporary features.yaml, extracts features, runs LOSO, and cleans up.
    """
    feature_subdir = f"preproc_{params.name}"
    temp_config_path = None
    
    try:
        # Create temporary config file
        temp_config_path = create_temp_features_yaml(params, features_yaml_path)
        
        with print_lock:
            logger.info(f"[{params.name}] Starting feature extraction...")
        
        # Extract features with this config
        success = extract_features(
            dataset, feature_subdir, config_path, 
            disable_gpu=disable_gpu,
            preprocess_config_path=temp_config_path,
            quiet=quiet
        )
        
        if not success:
            with print_lock:
                logger.error(f"[{params.name}] Feature extraction failed")
            return {
                'config_name': params.name,
                'bandpass_enabled': params.bandpass_enabled,
                'bandpass_lowcut': params.bandpass_lowcut,
                'bandpass_highcut': params.bandpass_highcut,
                'bandpass_order': params.bandpass_order,
                'notch_enabled': params.notch_enabled,
                'notch_freq': params.notch_freq,
                'notch_q': params.notch_q,
                'accuracy': None,
                'std': None,
                'f1': None,
                'n_folds': None,
                'error': 'extraction_failed',
            }
        
        with print_lock:
            logger.info(f"[{params.name}] Running LOSO evaluation...")
        
        # Run LOSO evaluation
        eval_results = run_loso_evaluation(dataset, model, feature_subdir, quiet=quiet)
        
        # Store results
        result = {
            'config_name': params.name,
            'bandpass_enabled': params.bandpass_enabled,
            'bandpass_lowcut': params.bandpass_lowcut,
            'bandpass_highcut': params.bandpass_highcut,
            'bandpass_order': params.bandpass_order,
            'notch_enabled': params.notch_enabled,
            'notch_freq': params.notch_freq,
            'notch_q': params.notch_q,
            # **eval_results,
        }
        
        with print_lock:
            if eval_results['accuracy'] is not None:
                logger.info(f"[{params.name}] Accuracy: {eval_results['accuracy']:.4f}")
            else:
                logger.warning(f"[{params.name}] Could not get accuracy")
        
        return result
        
    finally:
        # Clean up temp config
        if temp_config_path and temp_config_path.exists():
            os.unlink(temp_config_path)
        
        # Clean up features if not keeping them
        if not keep_features:
            feature_dir = Path(f"datasets/{dataset}/features/default/{feature_subdir}")
            if feature_dir.exists():
                shutil.rmtree(feature_dir)


def evaluate_preprocessing_configs(
    dataset: str,
    model: str,
    configs: List[PreprocessingParams],
    output_dir: Path,
    config_path: Path,
    features_yaml_path: Path,
    disable_gpu: bool = False,
    keep_features: bool = False,
    parallel: int = 1,
) -> List[dict]:
    """Evaluate multiple preprocessing configurations.
    
    Args:
        parallel: Number of parallel workers (1 = sequential)
    """
    results = []
    
    if parallel <= 1:
        # Sequential execution (original behavior)
        for i, params in enumerate(configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing config {i+1}/{len(configs)}: {params.name}")
            logger.info(f"{'='*60}")
            
            result = process_single_config(
                params=params,
                dataset=dataset,
                model=model,
                config_path=config_path,
                features_yaml_path=features_yaml_path,
                disable_gpu=disable_gpu,
                keep_features=keep_features,
                quiet=False,
            )
            results.append(result)
    else:
        # Parallel execution
        logger.info(f"\nRunning {len(configs)} configs with {parallel} parallel workers")
        logger.info("="*60)
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(
                    process_single_config,
                    params=params,
                    dataset=dataset,
                    model=model,
                    config_path=config_path,
                    features_yaml_path=features_yaml_path,
                    disable_gpu=disable_gpu,
                    keep_features=keep_features,
                    quiet=True,
                ): params
                for params in configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[{params.name}] Failed with exception: {e}")
                    results.append({
                        'config_name': params.name,
                        'bandpass_enabled': params.bandpass_enabled,
                        'bandpass_lowcut': params.bandpass_lowcut,
                        'bandpass_highcut': params.bandpass_highcut,
                        'bandpass_order': params.bandpass_order,
                        'notch_enabled': params.notch_enabled,
                        'notch_freq': params.notch_freq,
                        'notch_q': params.notch_q,
                        'accuracy': None,
                        'std': None,
                        'f1': None,
                        'n_folds': None,
                        'error': str(e),
                    })
    
    return results


def save_results(results: List[dict], output_path: Path) -> None:
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return
    
    # Sort by accuracy (descending)
    results_sorted = sorted(results, key=lambda x: x.get('accuracy') or 0, reverse=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(results_sorted[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_sorted)
    
    logger.info(f"Saved results to {output_path}")


def print_summary(results: List[dict]) -> None:
    """Print a summary of results."""
    if not results:
        print("\nNo results to display.")
        return
    
    # Filter out failed results for ranking
    valid_results = [r for r in results if r.get('accuracy') is not None]
    failed_results = [r for r in results if r.get('accuracy') is None]
    
    # Sort by accuracy
    results_sorted = sorted(valid_results, key=lambda x: x.get('accuracy') or 0, reverse=True)
    
    print("\n" + "="*80)
    print("PREPROCESSING EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Config Name':<25} {'Accuracy':>10} {'Std':>10} {'F1':>10}")
    print("-"*80)
    
    for r in results_sorted:
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] else "N/A"
        std = f"{r['std']:.4f}" if r['std'] else "N/A"
        f1 = f"{r['f1']:.4f}" if r['f1'] else "N/A"
        print(f"{r['config_name']:<25} {acc:>10} {std:>10} {f1:>10}")
    
    # Show failed configs
    if failed_results:
        print("-"*80)
        print("Failed configurations:")
        for r in failed_results:
            error = r.get('error', 'unknown error')
            print(f"  {r['config_name']}: {error}")
    
    print("="*80)
    
    # Best config details
    if results_sorted:
        best = results_sorted[0]
        print(f"\n{'*'*80}")
        print(f"BEST CONFIGURATION: {best['config_name']}")
        print(f"{'*'*80}")
        print(f"  Accuracy: {best['accuracy']:.4f}" + (f" ± {best['std']:.4f}" if best.get('std') else ""))
        if best.get('f1'):
            print(f"  F1 Score: {best['f1']:.4f}")
        print()
        print("  Preprocessing Settings:")
        if best['bandpass_enabled']:
            print(f"    Bandpass: {best['bandpass_lowcut']}-{best['bandpass_highcut']} Hz, order={best['bandpass_order']}")
        else:
            print(f"    Bandpass: disabled")
        if best['notch_enabled']:
            print(f"    Notch: {best['notch_freq']} Hz, Q={best['notch_q']}")
        else:
            print(f"    Notch: disabled")
        print()
        print("  To use these settings, update features.yaml:")
        print("    preprocessing:")
        print("      bandpass:")
        print(f"        enabled: {str(best['bandpass_enabled']).lower()}")
        print(f"        lowcut: {best['bandpass_lowcut']}")
        print(f"        highcut: {best['bandpass_highcut']}")
        print(f"        order: {best['bandpass_order']}")
        print("      notch:")
        print(f"        enabled: {str(best['notch_enabled']).lower()}")
        print(f"        freq: {best['notch_freq']}")
        print(f"        q_factor: {best['notch_q']}")
        print(f"{'*'*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate preprocessing parameter configurations for EMG classification"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="rami",
        help="Dataset to use (default: rami)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="lda",
        help="Model to use for evaluation (default: lda, fast and reliable)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick test with subset of configurations"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential). Use 4 for batch of 4."
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Disable GPU for feature extraction"
    )
    parser.add_argument(
        "--keep-features",
        action="store_true",
        help="Keep extracted features (don't clean up)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path (default: datasets/{dataset}/training/preprocessing_eval.csv)"
    )
    
    args = parser.parse_args()
    
    # Paths
    config_path = Path(f"datasets/{args.dataset}/config.yaml")
    features_yaml_path = Path("src/emg_classification/config/features.yaml")
    
    if not config_path.exists():
        logger.error(f"Dataset config not found: {config_path}")
        sys.exit(1)
    
    if not features_yaml_path.exists():
        logger.error(f"Features config not found: {features_yaml_path}")
        sys.exit(1)
    
    # Select configs
    configs = QUICK_CONFIGS if args.quick else PREPROCESSING_CONFIGS
    
    logger.info(f"Evaluating {len(configs)} preprocessing configurations")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")
    if args.parallel > 1:
        logger.info(f"Parallel workers: {args.parallel}")
    
    # Output path
    output_path = Path(args.output) if args.output else Path(f"datasets/{args.dataset}/training/preprocessing_eval.csv")
    
    # Run evaluation
    results = evaluate_preprocessing_configs(
        dataset=args.dataset,
        model=args.model,
        configs=configs,
        output_dir=output_path.parent,
        config_path=config_path,
        features_yaml_path=features_yaml_path,
        disable_gpu=args.disable_gpu,
        keep_features=args.keep_features,
        parallel=args.parallel,
    )
    
    # Save and display results
    # save_results(results, output_path)
    # print_summary(results)


if __name__ == "__main__":
    main()
