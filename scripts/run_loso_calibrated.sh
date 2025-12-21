#!/bin/bash
# LOSO evaluation with calibration and domain adaptation
# 
# Configuration rationale:
# - calibration 50: Subject adaptation using 50 samples/class from test subject
# - normalizer percentile: Domain adaptation to reduce inter-subject variability  
# - feat-select kbest k=100: Feature selection to reduce noise
# - subsample 0.1: Use 10% of data for faster iteration

MODELS="${1:-lda,rf-tuned,et-tuned,svm-tuned}"
OUTPUT="${2:-results/loso_calibrated.csv}"

cd "$(dirname "$0")/.." || exit 1

uv run python loso_train.py \
    --models "$MODELS" \
    --calibration 50 \
    --normalizer percentile \
    --feat-select kbest \
    --k 100 \
    --subsample 0.1 \
    --out-csv "$OUTPUT"
