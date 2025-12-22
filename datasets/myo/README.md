# GrabMyo Dataset

Surface EMG dataset for hand gesture recognition from PhysioNet.

## Overview

- **Reference**: [GrabMyo on PhysioNet](https://physionet.org/content/grabmyo/)
- **Subjects**: 43 healthy participants
- **Sessions**: 3 per subject (different days for cross-session evaluation)
- **Gestures**: 17 (16 active movements + rest)
- **Trials**: 7 per gesture per session
- **Sampling Rate**: 2048 Hz

## Electrode Configuration

The dataset uses two electrode bands with a total of 28 EMG channels:

### Forearm Band (channels 0-15)
- 16 channels
- Placed on the forearm extensors/flexors
- Captures primary muscle activity for most gestures

### Wrist Band (channels 16-27)
- 12 channels
- Placed around the wrist
- Captures tendon movement and distal muscle activity

## File Format

Data is stored in WFDB (WaveForm DataBase) format, the standard for PhysioNet:

- `.dat` files: Binary signal data
- `.hea` files: Header with metadata (channels, sampling rate, etc.)

### Filename Pattern

```
session{i}_participant{j}_gesture{k}_trial{l}
```

- `session`: 1, 2, or 3
- `participant`: 1-43
- `gesture`: 1-17
- `trial`: 1-7

## Gestures

| ID | Gesture |
|----|---------|
| 0/17 | Rest |
| 1 | Wrist Flexion |
| 2 | Wrist Extension |
| 3 | Wrist Pronation |
| 4 | Wrist Supination |
| 5 | Radial Deviation |
| 6 | Ulnar Deviation |
| 7 | Power Grip |
| 8 | Open Hand |
| 9 | Pinch Grip |
| 10 | Tripod Grip |
| 11 | Point Index |
| 12 | Thumb Up |
| 13 | Peace Sign |
| 14 | Rock Sign |
| 15 | OK Sign |
| 16 | Finger Spread |

## Usage

### Configuration Options

In `config.yaml`:

```yaml
dataset:
  loader_type: myo
  channel_group: forearm  # "forearm", "wrist", or "all"
  sessions: [1, 2, 3]     # Which sessions to include
```

### Channel Group Selection

- `"forearm"`: Use only forearm band (16 channels) - recommended for comparison with other datasets
- `"wrist"`: Use only wrist band (12 channels)
- `"all"`: Use all 28 channels for maximum information

### Cross-Session Evaluation

The three sessions per subject (recorded on different days) enable:

1. **Within-session**: Train/test on same session (higher accuracy)
2. **Cross-session**: Train on session(s), test on held-out session (realistic)
3. **Session adaptation**: Few-shot calibration with target session samples

## Data Download

```bash
# Using wget
wget -r -N -c -np https://physionet.org/files/grabmyo/1.0.2/

# Using PhysioNet API (requires account)
# See: https://physionet.org/content/grabmyo/
```

## Requirements

```bash
pip install wfdb  # For reading WFDB format
```

## Citation

If using this dataset, please cite:

```bibtex
@article{grabmyo2021,
  title={GrabMyo: A dataset of hand gestures recorded using surface EMG},
  author={...},
  journal={PhysioNet},
  year={2021}
}
```
