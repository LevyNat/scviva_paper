# Xenium human breast cancer (Fig. 4, S15-S17, Table 1)

Two segmentations of the same sample (S1, replicates 1 and 2): the original Xenium segmentation (`breast_original/`)
and a Proseg re-segmentation (`breast_proseg/`).

| step | command | output |
|---|---|---|
| 1. scANVI + scVIVA | `python runner.py` from `breast_original/` and from `breast_proseg/` (dedicated runners) | `checkpoints/xenium_breast_cancer_S1_R1_2/xenium_breast_cancer_S1_R1_2_nicheVI.h5ad`, `breast_proseg/checkpoints/proseg_xenium_breast_cancer_S1_R1_2/proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2_nicheVI.h5ad` |
| 2. Hotspot modules and DE of endothelial cells (Fig. 4C-G, S15) | `breast_original/hotspot_breast.ipynb`, `breast_proseg/hotspot_breast.ipynb` | figures in each `figures_journal/` |
| 3. two-step DE vs t-test (Fig. 4E, S16C) | `differential_expression_benchmark/t_test_xenium.ipynb`, `t_test_proseg.ipynb` | scatter plots in `differential_expression_benchmark/figures_2DE/` |
| 4. purity of the DE genes (Table 1) | `differential_expression_benchmark/benchmark_2DE.ipynb` (C-SIDE columns: `c_side_xenium.ipynb`, `c_side_xenium_import_weights.ipynb`, R kernel with `spacexr`) | purities against the scRNA-seq reference |
| 5. scRNA-seq markers (Fig. S17) | `breast_original/breast_scrna_DE.ipynb` | `breast_proseg/figures_journal/dotplot_scrna.svg` |

- The runners load the provided models when they exist. For the original segmentation, the embedding uses the 1001-epoch scVIVA model (`N_EPOCHS_NICHEVI`) and the DE notebooks use the 1000-epoch model (`N_EPOCHS_NICHEVI_DE`), as in the paper.
- scRNA-seq references: the Wu et al. 2021 atlas (Table 1; the DE notebooks build the endothelial reference `breast_sc_atlas_wu2021_DE.h5ad` from it) and the "Global Atlas" (Fig. S17), both from CELLxGENE (links in the main README).
- `preprocessing/` documents how the Proseg input was built (Proseg, scANVI and resolVI label transfer). We provide the final `proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad`.
