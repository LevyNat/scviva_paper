#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset> <gpu>}"
GPU="${2:?Usage: $0 <dataset> <gpu>}"

ROOT="/home/nathanl/scviva_paper/baselines"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATASET}_$(date +%Y%m%d_%H%M%S).log"

# SimVI's forward pass runs the full graph every iteration regardless of batch_size,
# so large datasets (e.g. cosmx_cortex) sit right at the GPU memory ceiling -
# expandable_segments reduces allocator fragmentation overhead and costs nothing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# conda's own activation hooks (e.g. libblas_mkl_activate.sh) reference variables
# like $MKL_INTERFACE_LAYER without a default, which is fine under normal bash but
# fatal under `set -u` — relax it only around conda sourcing/activation.
set +u
source /home/nathanl/miniforge3/etc/profile.d/conda.sh
set -u

{
  echo "==> [$DATASET] Running SimVI (env: simvi24, gpu: $GPU)"
  set +u
  conda activate simvi24
  set -u
  cd "$ROOT/SIMVI"
  python simvi_runner_2025.py --dataset "$DATASET" --gpu "$GPU"
  set +u
  conda deactivate
  set -u

  echo "==> [$DATASET] Running BANKSY (env: scvi)"
  set +u
  conda activate scvi
  set -u
  cd "$ROOT/BANKSY"
  python banksy_script.py --dataset "$DATASET"
  set +u
  conda deactivate
  set -u

  echo "==> [$DATASET] Done."
} 2>&1 | tee "$LOG_FILE"
