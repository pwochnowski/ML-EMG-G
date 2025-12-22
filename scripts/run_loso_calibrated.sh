#!/bin/bash
# LOSO evaluation with calibration and domain adaptation
# 
# Usage: run_loso_calibrated.sh [DATASET] [MODELS] [-s]
#
# Configuration rationale:
# - calibration 50: Subject adaptation using 50 samples/class from test subject
# - normalizer percentile: Domain adaptation to reduce inter-subject variability  
# - feat-select kbest k=100: Feature selection to reduce noise
# - subsample 0.1: Use 10% of data for faster iteration

DATASET="${1:-rami}"
MODELS="${2:-lda,rf-tuned,et-tuned,svm-tuned}"
PERSIST="${3:-}"

cd "$(dirname "$0")/.." || exit 1

# Build output args only if --persist flag is set
OUTPUT_ARG=""
if [ "$PERSIST" = "-s" ]; then
    OUTPUT_ARG="--out-csv datasets/${DATASET}/training/loso_calibrated.csv"
fi

uv run python loso_train.py \
    --dataset "$DATASET" \
    --models "$MODELS" \
    --calibration 50 \
    --normalizer percentile \
    --feat-select kbest \
    --k 100 \
    --subsample 0.1 \
    $OUTPUT_ARG
