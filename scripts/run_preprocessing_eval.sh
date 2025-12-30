#!/bin/bash
# Run preprocessing evaluation across all datasets
#
# Usage:
#   ./scripts/run_preprocessing_eval.sh           # Quick test (4 configs per dataset)
#   ./scripts/run_preprocessing_eval.sh --full    # Full test (13 configs per dataset)
#
# Results will be saved to datasets/{dataset}/training/preprocessing_eval.csv

set -e

# Parse arguments
QUICK_FLAG="--quick"
PARALLEL=1
MODEL="lda"

if [[ "$1" == "--full" ]]; then
    QUICK_FLAG=""
    echo "Running FULL preprocessing evaluation (13 configs per dataset)"
else
    echo "Running QUICK preprocessing evaluation (4 configs per dataset)"
    echo "Use --full for complete evaluation"
fi

echo "Model: $MODEL, Parallel workers: $PARALLEL"
echo ""

DATASETS=(
    #"rami"
    # "myo" 
    "db1"
)
TOTAL_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo "========================================================================"
    echo "Evaluating preprocessing for: $dataset"
    echo "========================================================================"
    
    START=$(date +%s)
    
    uv run python evaluate_preprocessing.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --parallel "$PARALLEL" \
        --keep-features \
        $QUICK_FLAG
    
    END=$(date +%s)
    DURATION=$((END - START))
    echo ""
    echo "[$dataset] Completed in ${DURATION}s"
    echo "Results saved to: datasets/$dataset/training/preprocessing_eval.csv"
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "========================================================================"
echo "ALL DATASETS COMPLETE"
echo "========================================================================"
echo "Total time: ${TOTAL_DURATION}s"
echo ""
echo "Summary of results:"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "--- $dataset ---"
    if [[ -f "datasets/$dataset/training/preprocessing_eval.csv" ]]; then
        # Print header and top 3 results
        head -n 4 "datasets/$dataset/training/preprocessing_eval.csv" | column -t -s,
    else
        echo "  No results found"
    fi
    echo ""
done
