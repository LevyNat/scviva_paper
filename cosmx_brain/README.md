# CosMx human frontal cortex (Fig. S9)

**Benchmark (Fig. S9A-C)**

| step | command | output |
|---|---|---|
| 1. build the data | `cosmx_cortex.ipynb`: reads the CosMx flat files, writes `cosmx_cortex.h5ad`; cell 9 selects 5,000 HVGs (seurat_v3 on raw counts) and writes `cosmx_cortex_hvg5000.h5ad`; the following cells define the tissue regions (KMeans on the niche composition) | `cosmx_cortex_hvg5000.h5ad` |
| 2. baselines | `(cd baselines && ./run_benchmark.sh cosmx_cortex_hvg5000 0)` | BANKSY and SimVI embeddings, written into the data file |
| 3. scANVI + scVIVA + scIB | `python runner.py cosmx_brain` (from the repo root) | scIB tables and plots in `cosmx_brain/figures/` |

**Duplicated astrocytes (Fig. S9D-F)**: astro.2 astrocytes from the upper cortical layers (L2/3) duplicated into the vessel / peri-vascular region.

| step | command | output |
|---|---|---|
| 1. build the input | `duplicate_analysis.ipynb` (`run_tag = "astro.2_upper_vessel"`) | `cosmx_cortex_astro.2_upper_vessel_replaced.h5ad` |
| 2. baselines | `(cd baselines && ./run_benchmark.sh cosmx_cortex_astro.2_upper_vessel_replaced 0)` | BANKSY and SimVI embeddings |
| 3. scANVI + scVIVA | `DUP_CELL_TYPE=astro.2_upper_vessel python runner.py cosmx_brain/duplicate` | `cosmx_brain/checkpoints/cosmx_cortex_astro.2_upper_vessel_replaced/` |
| 4. figures | `duplicate_analysis.ipynb`, after training | UMAPs and iLISI distributions |