#!/bin/bash
# Run robustness experiments with multiple seeds and aggregate results
#
# Usage:
#   ./run_experiment.sh              # Run everything (main + ablation + aggregation)
#   ./run_experiment.sh --ablation   # Run ablation only + aggregation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNNER_ARGS=""
MODE="Full"

if [ "$1" = "--ablation" ]; then
    RUNNER_ARGS="--ablation-only"
    MODE="Ablation Only"
fi

echo "=============================================="
echo "Starting Robustness Experiment ($MODE)"
echo "=============================================="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Run training and benchmarking
echo "Step 1: Running runner.py $RUNNER_ARGS"
echo "----------------------------------------------"
python runner.py $RUNNER_ARGS

echo ""
echo "=============================================="
echo "Step 1 Complete: Training and benchmarking done"
echo "=============================================="
echo ""

# Step 2: Aggregate results and generate plots
echo "Step 2: Running aggregate_results.py"
echo "----------------------------------------------"
python aggregate_results.py

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results saved to: /home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/seeds/"
