# Ablations and hyperparameter sensitivity: MERFISH mouse brain (Fig. S3)

scVIVA is retrained with one change at a time, with the default (K = 20 neighbors, spatial weight β = 10) as reference:

| configuration (`params.py`) | change |
|---|---|
| `eta0_*` | no niche state (η) decoder |
| `s0_*` | no niche decoders (no η, no α) |
| `s5_*`, `s20_*` | β = 5, β = 20 |
| `k10_s10_*`, `k50_s10_*`, `k100_s10_*` | K = 10, 50, 100 |

| step | command | output |
|---|---|---|
| 1. train and score | `python runner.py merfish_brain/ablation` (from the repo root) | `merfish_brain/figures/revisions_benchmark/`: `scib_cell_type_results_adata_M1_M2_core_6_sections_ablation_k.csv` (cell type preservation, batch correction), `BioCons_cell_type_..._ablation_k.csv` (region preservation per cell type) |
| 2. plot | `merfish_brain/Figure_S2_S3.ipynb`, section "Fig. S3" | `ablation_k.svg` |

The models are saved in `merfish_brain/checkpoints/adata_M1_M2_core_6_sections/`; the runner loads them when they exist.
