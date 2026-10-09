# Visium HD colorectal cancer (Fig. S1B, S6, S7)

**Data and annotation (Fig. S6A, S7)**

| step | command | output |
|---|---|---|
| 1. cells from the Space Ranger segmentation | `crc_analysis.ipynb`, first part | `adata.h5ad`, `adata_recovered.h5ad` |
| 2. resolVI | `python resolvi.py` | `adata_resolvi.h5ad` (resolVI embedding and Leiden clusters); the model is saved in `resolvi_ckpts/` and reloaded when present |
| 3. annotation, regions, gene selection | `crc_analysis.ipynb`, from "adata_resolvi": cell types from the Leiden clusters, regions by KMeans on the niche composition, 4,000 HVGs | `adata_legacy_hvg4k.h5ad` |

**Benchmark (Fig. S6B-C)**

| step | command | output |
|---|---|---|
| 1. baselines | `(cd baselines && ./run_benchmark.sh crc_visium_hd 0)` | BANKSY and SimVI embeddings, written into the data file |
| 2. scANVI + scVIVA + scIB | `python runner.py crc_visium_hd` (from the repo root) | `crc_visium_hd/figures/benchmark/` and `crc_visium_hd/checkpoints/adata_legacy_hvg4k/adata_legacy_hvg4k_nicheVI.h5ad` |
| 3. figures | `cell_type_DE.ipynb` (region vs cell type preservation), `crc_analysis.ipynb` last part (UMAPs, spatial maps) | `figures/` |

**Memory and runtime (Fig. S6D)**: `python memory_sweep.py` (scVIVA), `baselines/SIMVI/memory_sweep.py`, `baselines/BANKSY/benchmark_memory.py`. They read `adata_legacy_hvg{1,2,3,4}k.h5ad` (1,000-4,000 HVGs) and write to `crc_visium_hd/figures/memory/`.

**Duplicated macrophages (Fig. S6E-G)**

| step | command | output |
|---|---|---|
| 1. build the input | `duplicate_analysis.ipynb`, first part | `adata_Macrophages_replaced.h5ad` |
| 2. baselines | `(cd baselines && ./run_benchmark.sh crc_visium_hd_macrophages_replaced 0)` | BANKSY and SimVI embeddings |
| 3. scANVI + scVIVA | `python runner.py crc_visium_hd/duplicate` | `crc_visium_hd/checkpoints/adata_Macrophages_replaced/adata_Macrophages_replaced_nicheVI.h5ad` |
| 4. figures | `duplicate_analysis.ipynb`, after training | UMAPs, iLISI |

**Robustness to annotation errors (Fig. S1B)**: see `cell_type_robustness/robustness_to_annotation_error/README.md`.
