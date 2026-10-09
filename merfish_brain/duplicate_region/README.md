# Duplicated astrocytes: MERFISH mouse brain (Fig. 2D-F, S4)

Astrocytes of the isocortex are replaced by duplicates of astrocytes from the thalamus and hypothalamus, placed at
the isocortex positions: same expression, different niche. 

| step | command | output |
|---|---|---|
| 1. build the input | `duplicate_analysis.ipynb`, section "Building the replaced adata object" | `adata_astrocyte_replaced.h5ad` |
| 2. baselines | `(cd baselines && ./run_benchmark.sh brain_merfish_astrocyte_replaced 0)` | BANKSY and SimVI embeddings, written into the input file |
| 3. scANVI + scVIVA | `python runner.py merfish_brain/duplicate_region` (from the repo root) | `merfish_brain/checkpoints/adata_astrocyte_replaced/adata_astrocyte_replaced_nicheVI.h5ad` |
| 4. figures | `duplicate_analysis.ipynb`, from "results after training" | UMAPs, iLISI distributions, reconstruction errors (Fig. S4B), in `figures/` |
