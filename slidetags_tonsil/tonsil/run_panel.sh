#!/usr/bin/env bash
# Full tonsil benchmark for one gene panel, e.g.:  ./run_panel.sh 15000hvg
#   RES_RULE=target ./run_panel.sh 10000hvg     -> target 14-16 Leiden clusters instead of the silhouette rule
# Needs /home/nathanl/Data/slidetags/tonsil_nl_<panel>.h5ad (slidetags_build.ipynb), a
# slidetags_tonsil_<panel> entry in baselines/dataset_params.py, and
# slidetags_tonsil/tonsil_<panel>/params.py. GPU 1 = unlabeled arm, GPU 0 = everything else.
set -eo pipefail
PANEL="${1:?usage: $0 <panel, e.g. 15000hvg>}"
REPO=/home/nathanl/scviva_paper
PY=/home/nathanl/miniforge3/envs/scvi/bin/python
LOG=$REPO/slidetags_tonsil/logs_nl_panels; mkdir -p "$LOG"
export JAX_PLATFORMS=cpu   # scib's JAX: CPU is enough and avoids competing with torch for GPU memory

# 1) unlabeled arm (scVI -> Leiden pseudo-labels -> unlabeled scVIVA -> scib), GPU 1
# resolution rule for the pseudo-labels: RES_RULE=internal (default; label-free: silhouette, Calinski-Harabasz and
# Davies-Bouldin on the 0.05..2.00 grid, majority vote, silhouette breaks ties) or RES_RULE=target (14-16 clusters)
if [ "${RES_RULE:-internal}" = internal ]; then RES_ENV="NL_RES_SELECT=internal"; else RES_ENV="NL_TARGET_CLUSTERS=14,16"; fi
( cd $REPO/slidetags_tonsil/tonsil && env CUDA_VISIBLE_DEVICES=1 NL_DATA_FILE=tonsil_nl_$PANEL.h5ad \
    $RES_ENV NL_REGION=1 $PY runner_no_labels.py > "$LOG/unlabeled_$PANEL.log" 2>&1 ) &
NL_PID=$!
# let it finish reading the data file before BANKSY/SimVI start writing embeddings into it
until grep -q "Will save to" "$LOG/unlabeled_$PANEL.log" 2>/dev/null; do
    kill -0 $NL_PID 2>/dev/null || { echo "unlabeled arm died early"; tail "$LOG/unlabeled_$PANEL.log"; exit 1; }
    sleep 5
done

# 2) SimVI + BANKSY (write into the data file), then supervised scANVI + scVIVA + runner scib, GPU 0
( cd $REPO/baselines && ./run_benchmark.sh slidetags_tonsil_$PANEL 0 )
( cd $REPO && CUDA_VISIBLE_DEVICES=0 $PY runner.py slidetags_tonsil/tonsil_$PANEL > "$LOG/supervised_$PANEL.log" 2>&1 )

# 3) wait for the unlabeled arm, then one table/scatter with every method
wait $NL_PID
( cd $REPO/slidetags_tonsil/tonsil && $PY score_panel.py $PANEL 2>&1 | tee "$LOG/score_$PANEL.log" )
( cd $REPO/slidetags_tonsil/tonsil && $PY plot_scatter.py $PANEL --drop PCA )  # Fig. S9C: the scatter without PCA
echo "PANEL_DONE $PANEL"
