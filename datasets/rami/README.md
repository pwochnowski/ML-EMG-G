# Rami EMG Dataset Documentation

## Overview

The Rami dataset contains surface EMG recordings of **8 hand gestures** from **11 subjects** across **5 arm positions**. It is designed for studying position-invariant EMG classification.

- **Sampling Rate**: 1000 Hz
- **EMG Channels**: 7 channels
- **Subjects**: 11 (10 male, 1 female)
- **Gestures**: 8 classes
- **Positions**: 5 arm positions

## Data Organization

```
datasets/rami/data/
├── S1_Male/
│   ├── Pos1_WristFlex_1.txt
│   ├── Pos1_WristFlex_2.txt
│   ├── Pos1_WristExte_1.txt
│   └── ...
├── S2_Male/
│   └── ...
└── S11_Male/
```

Each text file contains:
- Whitespace-separated columns (7 EMG channels)
- Each row is one time sample (1ms at 1000Hz)
- Filename format: `Pos{N}_{Gesture}_{Rep}.txt`

## Gestures (8 classes)

| Label | Gesture | Description |
|-------|---------|-------------|
| 1 | WristFlex | Wrist flexion |
| 2 | WristExte | Wrist extension |
| 3 | WristPron | Wrist pronation |
| 4 | WristSupi | Wrist supination |
| 5 | ObjectGri | Object grip |
| 6 | PichGrip_ | Pinch grip |
| 7 | HandOpen_ | Hand open |
| 8 | HandRest_ | Hand rest |

## Arm Positions (5 positions)

| Position | Description |
|----------|-------------|
| Pos1 | Neutral position |
| Pos2 | Arm raised |
| Pos3 | Arm extended forward |
| Pos4 | Arm to the side |
| Pos5 | Arm lowered |

## Subjects

| Subject | Gender | Notes |
|---------|--------|-------|
| S1_Male | Male | |
| S2_Male | Male | |
| S3_Male | Male | |
| S4_Male | Male | |
| S5-Male | Male | Note: hyphen in name |
| S6_Male | Male | |
| S7_Male | Male | |
| S8_Male | Male | |
| S9_Female | Female | |
| S10_Male | Male | |
| S11_Male | Male | |

## Recommended Preprocessing

1. **Window size**: 400ms (400 samples at 1000Hz) with 75% overlap
2. **Features**: Combined spectral (Khushaba) + time-domain features
3. **Normalization**: Standard scaling

## Usage

```bash
# Extract features for all subjects
uv run emg-extract --config datasets/rami/config.yaml --out-dir results/rami/

# Extract for specific subjects
uv run emg-extract --config datasets/rami/config.yaml S1_Male S2_Male --out-dir results/rami/

# Filter by position
uv run emg-extract --config datasets/rami/config.yaml --positions Pos1 Pos2 --out-dir results/rami/
```

## Key Challenges

1. **Position variance**: EMG patterns change with arm position
2. **Cross-position generalization**: Models trained on one position may not generalize
3. **Leave-one-subject-out (LOSO)**: Standard evaluation protocol
