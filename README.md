# EMG Classification Pipeline

A Python toolkit for EMG (electromyography) signal classification using machine learning. Supports feature extraction, Leave-One-Subject-Out (LOSO) cross-validation, and hyperparameter tuning for various classifiers.

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Install uv** 

2. **Clone and set up the project**:
   ```bash
   git clone <repository-url>
   cd rami_playground
   uv sync
   ```

3. **Install optional dependencies** (for XGBoost, LightGBM, etc.):
   ```bash
   uv sync --extra boosting    # XGBoost, LightGBM
   uv sync --extra dev         # pytest, ruff
   uv sync --extra all         # Everything
   ```

## Data Setup

**Important**: Raw EMG data must be manually placed into the appropriate dataset directories.

### Directory Structure

```
datasets/
├── {dataset_name}/
│   ├── config.yaml      # Pipeline configuration
│   ├── data/            # ⚠️ Raw data files (MANUAL PLACEMENT REQUIRED)
│   ├── features/        # Extracted features (.npz)
│   ├── training/        # Training results (.csv)
│   ├── tuning/          # Hyperparameter tuning results
│   ├── models/          # Saved models (.joblib)
│   └── analysis/        # Analysis outputs
```

### NinaPro DB1

1. Download from https://ninapro.hevs.ch/
2. Extract to `datasets/db1/data/`
3. Expected structure:
   ```
   datasets/db1/data/
   ├── s01/
   │   ├── S1_A1_E1.mat
   │   ├── S1_A1_E2.mat
   │   └── S1_A1_E3.mat
   ├── s02/
   │   └── ...
   └── s27/
   ```

### Rami Dataset

1. Copy or symlink data to `datasets/rami/data/`
2. Expected structure:
   ```
   datasets/rami/data/
   ├── S1_Male/
   │   ├── Pos1_WristFlex_1.txt
   │   └── ...
   ├── S2_Male/
   │   └── ...
   ```

### GrabMyo Dataset

1. Place data in `datasets/myo/data/`
2. See `datasets/myo/README.md` for specific structure

## Feature Extraction

Extract features from raw EMG data before training. Features are saved as `.npz` files.

### Using the CLI (recommended)

```bash
# Extract features for all subjects in a dataset
uv run emg-extract --config datasets/rami/config.yaml --all-subjects

# Extract for a specific subject
uv run emg-extract --config datasets/rami/config.yaml --subject S1_Male

# Specify feature type
uv run emg-extract --config datasets/db1/config.yaml --all-subjects --features spectral
uv run emg-extract --config datasets/db1/config.yaml --all-subjects --features time
uv run emg-extract --config datasets/db1/config.yaml --all-subjects --features combined
```

### Using standalone scripts

```bash
# Extract time-domain features from GrabMyo dataset
uv run extract_myo_features.py --channel-group forearm

# Custom window parameters
uv run extract_myo_features.py --window-size 256 --overlap 128 --subjects s01 s02 s03
```

### Combining Features

Combine spectral and time-domain features into a single file:

```bash
# Using dataset config (auto-resolves paths)
uv run combine_features.py --dataset rami --subject S1_Male

# Or specify files explicitly
uv run combine_features.py \
    --spec datasets/rami/features/S1_Male_features_khushaba.npz \
    --time datasets/rami/features/S1_Male_features_time.npz \
    --out datasets/rami/features/S1_Male_features_combined.npz
```

## Training with LOSO Cross-Validation

Leave-One-Subject-Out (LOSO) evaluates model generalization across subjects by training on N-1 subjects and testing on the held-out subject.

### Basic Usage

```bash
# Train LDA and SVM with LOSO evaluation
uv run loso_train.py --dataset rami --models lda,svm

# Specify models to evaluate
uv run loso_train.py --dataset db1 --models lda,svm,knn,rf,et
```

### Available Models

| Model | Name | Description |
|-------|------|-------------|
| `lda` | Linear Discriminant Analysis | Fast baseline |
| `svm` | Support Vector Machine (RBF) | Good accuracy |
| `knn` | K-Nearest Neighbors | Simple, no training |
| `rf` | Random Forest | Ensemble method |
| `et` | Extra Trees | Fast ensemble |
| `xgb` | XGBoost | Gradient boosting |
| `lgbm` | LightGBM | Fast gradient boosting |

### Advanced Options

```bash
# Feature selection (K-best features)
uv run loso_train.py --dataset rami --models svm --feat-select kbest --k 50

# PCA dimensionality reduction
uv run loso_train.py --dataset rami --models svm --feat-select pca --pca-n 30

# Subsample data for faster iteration
uv run loso_train.py --dataset rami --models lda,svm --subsample 0.1

# Per-position evaluation (for position-variant datasets)
uv run loso_train.py --dataset rami --models svm --per-position

# Save trained models
uv run loso_train.py --dataset rami --models et --save-models-dir datasets/rami/models

# With calibration (use N samples from test subject)
uv run loso_train.py --dataset rami --models svm --calibration 5

# Within-subject CV instead of LOSO
uv run loso_train.py --dataset rami --models svm --within-subject --n-folds 5

# Different normalization strategies
uv run loso_train.py --dataset rami --models svm --normalizer percentile
```

### Output

Results are saved to `datasets/{dataset}/training/loso_summary.csv`:

```csv
model,mean_acc,std_acc,n_folds
lda,0.7234,0.0512,11
svm,0.8156,0.0423,11
```

## Hyperparameter Tuning

Tune model hyperparameters using grid search with LOSO cross-validation:

```bash
# Run all tuners
uv run tune_all.py --dataset rami

# Quick tuning (smaller parameter grids)
uv run tune_all.py --dataset rami --quick

# Tune specific models
uv run tune_svm.py --dataset rami
uv run tune_knn.py --dataset rami
uv run tune_rf.py --dataset rami
uv run tune_xgb.py --dataset rami   # requires xgboost
uv run tune_lgbm.py --dataset rami  # requires lightgbm
```

Results are saved to `datasets/{dataset}/tuning/`.

## Project Structure

```
.
├── README.md                 # This file
├── pyproject.toml           # Project dependencies
├── datasets/                # Dataset directories
│   ├── db1/                 # NinaPro DB1
│   ├── rami/                # Rami dataset
│   └── myo/                 # GrabMyo dataset
├── src/emg_classification/  # Core library
│   ├── cli/                 # Command-line interfaces
│   ├── config/              # Configuration handling
│   ├── data/                # Data loaders
│   ├── features/            # Feature extraction
│   └── models/              # Model definitions
├── models.py                # Model factory
├── loso_train.py           # LOSO training script
├── combine_features.py     # Feature combination
├── extract_myo_features.py # GrabMyo feature extraction
├── tune_*.py               # Hyperparameter tuning scripts
└── tests/                  # Unit tests
```

## Running Tests

```bash
uv run pytest
```

## Configuration

Each dataset has a `config.yaml` file that specifies:

- Sampling rate and channel configuration
- Subject list
- Output directories
- Feature extraction parameters

Example (`datasets/rami/config.yaml`):
```yaml
dataset:
  name: rami
  n_channels: 7
  sampling_rate: 1000.0
  loader_type: rami
  subjects:
    S1_Male: s01
    S2_Male: s02
    # ...

output:
  features_dir: features
  training_dir: training
  models_dir: models
```

## Quick Start Example

```bash
# 1. Set up the project
uv sync

# 2. Place your data in datasets/rami/data/

# 3. Extract features for all subjects
uv run emg-extract --config datasets/rami/config.yaml --all-subjects --features combined

# 4. Train and evaluate with LOSO
uv run loso_train.py --dataset rami --models lda,svm,rf

# 5. (Optional) Tune hyperparameters
uv run tune_all.py --dataset rami --quick
```

## License

See LICENSE file for details.
