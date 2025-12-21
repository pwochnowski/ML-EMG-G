# NinaPro Database 1 (DB1) Dataset Documentation

## Overview

NinaPro DB1 contains surface EMG recordings from **27 intact subjects** performing **52 different hand/wrist movements** across 3 exercises. It is one of the most widely used benchmarks for EMG-based gesture recognition.

- **Sampling Rate**: 100 Hz
- **EMG Channels**: 10 channels
- **Subjects**: 27 (intact)
- **Total Movements**: 52 (+ rest class)
- **Repetitions**: 10 per movement per subject

## Data Organization

Each subject has 3 `.mat` files, one per exercise:
```
datasets/ninapro_db1/data/
├── s01/
│   ├── S1_A1_E1.mat  # Exercise 1
│   ├── S1_A1_E2.mat  # Exercise 2
│   └── S1_A1_E3.mat  # Exercise 3
├── s02/
│   └── ...
└── s27/
```

## Exercises

### Exercise 1: Basic Finger Movements (12 movements, labels 1-12)
| ID | Movement |
|----|----------|
| 1-2 | Index flexion and extension |
| 3-4 | Middle flexion and extension |
| 5-6 | Ring flexion and extension |
| 7-8 | Little finger flexion and extension |
| 9-10 | Thumb adduction and abduction |
| 11-12 | Thumb flexion and extension |

### Exercise 2: Hand Postures + Wrist Movements (17 movements, labels 13-29)

**Hand Postures (13-20):**
| Global ID | Movement |
|-----------|----------|
| 13 | Thumb up |
| 14 | Flexion of ring and little finger; thumb flexed over middle and little |
| 15 | Flexion of ring and little finger |
| 16 | Thumb opposing base of little finger |
| 17 | Abduction of the fingers |
| 18 | Fingers flexed together |
| 19 | Pointing index |
| 20 | Fingers closed together |

**Wrist Movements (21-29):**
| Global ID | Movement |
|-----------|----------|
| 21-22 | Wrist supination and pronation (rotation axis through middle finger) |
| 23-24 | Wrist supination and pronation (rotation axis through little finger) |
| 25-26 | Wrist flexion and extension |
| 27-28 | Wrist radial and ulnar deviation |
| 29 | Wrist extension with closed hand |

### Exercise 3: Grasping and Functional Movements (23 movements, labels 30-52)
| Global ID | Movement |
|-----------|----------|
| 30-31 | Large and small diameter grasp |
| 32 | Fixed hook |
| 33 | Index finger extension |
| 34 | Medium wrap |
| 35 | Ring grasp |
| 36 | Prismatic four fingers |
| 37 | Stick grasp |
| 38 | Writing tripod |
| 39-41 | Power, three finger, and precision sphere |
| 42 | Tripod |
| 43-44 | Prismatic and tip pinch |
| 45 | Quadpod |
| 46 | Lateral grasp |
| 47-48 | Parallel extension and flexion |
| 49 | Power disk |
| 50 | Open a bottle with a tripod grasp |
| 51 | Turn a screw (stick grasp) |
| 52 | Cut something (index finger extension grasp) |

## MATLAB File Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `subject` | (1, 1) | Subject number |
| `exercise` | (1, 1) | Exercise number (1, 2, or 3) |
| `emg` | (N, 10) | Surface EMG signal |
| `glove` | (N, 22) | Cyberglove sensor data (uncalibrated) |
| `stimulus` | (N, 1) | Movement label (as displayed in video) |
| `restimulus` | (N, 1) | Movement label (refined timing, **use this**) |
| `repetition` | (N, 1) | Repetition number for stimulus |
| `rerepetition` | (N, 1) | Repetition number for restimulus |

### EMG Channel Layout
- **Channels 1-8**: Electrodes equally spaced around the forearm at the height of the radio-humeral joint
- **Channel 9**: Main activity spot of flexor digitorum superficialis
- **Channel 10**: Main activity spot of extensor digitorum superficialis

### Label Notes
- **Stimulus 0** = Rest (present between all movements)
- **restimulus** is preferred over `stimulus` as it has refined timing that better corresponds to actual movement onset/offset
- **10 repetitions** per movement per subject

## Global Label Mapping

When combining all exercises, labels are offset to be globally unique:

| Exercise | Original Labels | Global Labels | Description |
|----------|-----------------|---------------|-------------|
| Rest | 0 | 0 | Rest class (same across all) |
| E1 | 1-12 | 1-12 | Finger movements |
| E2 | 1-17 | 13-29 | Hand postures + wrist |
| E3 | 1-23 | 30-52 | Grasping/functional |

**Total: 52 movement classes + rest**

The `MatFileLoader` applies this offset automatically when `global_labels=True` (default).

## Recommended Preprocessing

1. **Use `restimulus`** for labels (refined timing)
2. **Window size**: 200ms (20 samples at 100Hz) with 50-75% overlap
3. **Channel selection**: 
   - Forearm only: channels 1-8
   - All channels: channels 1-10
4. **Rest class**: Include (label 0) or exclude depending on application

## Usage

```bash
# Extract features for all subjects
uv run emg-extract --config datasets/ninapro_db1/config.yaml --out-dir results/ninapro/

# Extract for specific subjects
uv run emg-extract --config datasets/ninapro_db1/config.yaml --subjects s01 s02 --out-dir results/ninapro/

# Filter by exercise
uv run emg-extract --config datasets/ninapro_db1/config.yaml --exercises 1 --out-dir results/ninapro/
```

## References

- Atzori, M., Gijsberts, A., et al. (2014). Electromyography data for non-invasive naturally-controlled robotic hand prostheses. *Scientific Data*.
- Gijsberts, A., et al. (2014). Movement error rate for evaluation of machine learning methods for sEMG-based hand movement classification. *IEEE TNSRE*.
- Dataset: https://ninapro.hevs.ch/
