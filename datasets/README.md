# EMG Datasets

This directory contains EMG datasets for gesture recognition research.

## Available Datasets

| Dataset | Subjects | Channels | Classes | Sampling Rate | Description |
|---------|----------|----------|---------|---------------|-------------|
| [db1](db1/) | 27 | 10 | 52 | 100 Hz | NinaPro DB1 - Hand movements |
| [rami](rami/) | 11 | 7 | 8 | 1000 Hz | Position-variant gestures |

## Directory Structure

Each dataset is self-contained with the following structure:
```
datasets/
├── {dataset_name}/
│   ├── README.md        # Dataset documentation
│   ├── config.yaml      # Pipeline configuration
│   ├── data/            # Raw data files
│   ├── features/        # Extracted features (.npz)
│   ├── training/        # Training results (.csv summaries)
│   ├── tuning/          # Hyperparameter tuning results
│   ├── models/          # Saved model files (.joblib)
│   └── analysis/        # Analysis notebooks and reports
```

## Setup Instructions

### NinaPro DB1

1. Download from https://ninapro.hevs.ch/
2. Extract to `datasets/db1/data/`
3. Structure should be:
   ```
   datasets/db1/data/
   ├── s01/
   │   ├── S1_A1_E1.mat
   │   ├── S1_A1_E2.mat
   │   └── S1_A1_E3.mat
   ├── s02/
   └── ...
   ```

### Rami Dataset

1. Copy or symlink data to `datasets/rami/data/`
2. Structure should be:
   ```
   datasets/rami/data/
   ├── S1_Male/
   │   ├── Pos1_WristFlex_1.txt
   │   └── ...
   ├── S2_Male/
   └── ...
   ```

## Quick Start

```bash
# Extract features (output goes to datasets/{dataset}/features/)
uv run python pipeline.py datasets/rami/data/S1_Male --dataset rami --feat khushaba
uv run python pipeline.py datasets/rami/data/S1_Male --dataset rami --feat time

# Combine spectral + time features
uv run python combine_features.py --dataset rami --subject S1_Male

# Train models (reads from datasets/{dataset}/features/, writes to datasets/{dataset}/training/)
uv run python loso_train.py --dataset rami --models lda,svm

# Run hyperparameter tuning (writes to datasets/{dataset}/tuning/)
uv run python tune_all.py --dataset rami --quick
```

## Adding New Datasets

1. Create `datasets/{name}/` directory
2. Add `README.md` with dataset documentation
3. Add `config.yaml` with pipeline configuration (include `output:` section)
4. Place raw data in `data/` subdirectory
5. Implement a loader in `src/emg_classification/data/loaders/` if needed
