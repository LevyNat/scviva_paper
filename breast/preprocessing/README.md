# How the Proseg input was built (Fig. 4B right, every Proseg panel)

Goal: give the cells of the Proseg re-segmentation the cell types of the original Xenium segmentation.
The result is `proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad`, the only file the Proseg analysis
(`breast/breast_proseg/`) reads. It uses two of its columns: `predictions_resolvi_proseg_coarse_corrected`
(cell types) and `kmeans_alpha_5` (niches).

**These files document the procedure and we provide the final result
`proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad`.

| step | file | what it does | adds |
|---|---|---|---|
| 1 | `1_build_proseg_anndata.ipynb`, cells 0-33 | reads the Proseg outputs of replicates R1 and R2 (expected counts, transcript metadata), keeps the genes of the original panel, filters cells, concatenates the replicates | `proseg_xenium_breast_cancer_S1_R1_2.h5ad` |
| 2 | `2_scanvi_transfer.py` | scANVI reference on the original segmentation (author cell types), then query mapping of the Proseg cells | `predictions_scanvi`, `X_scANVI` |
| 2b | `1_build_proseg_anndata.ipynb`, from cell 35 | copies the scANVI predictions onto the Proseg cells | `proseg_scanvi_xenium_breast_cancer_S1_R1_2.h5ad` |
| 3 | `3_resolvi_transfer.py` | resolVI reference on the original segmentation, then query mapping of the Proseg cells | `predicted_celltype_resolvi`, `X_resolvi` → `proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad` |
| 4 | `4_resolvi_de_novo.py` | cells where scANVI and resolVI disagree are left unlabeled; a semi-supervised resolVI trained on the Proseg cells assigns them | `predictions_resolvi_proseg`, `X_resolvi_proseg` |
| 5 | `5_coarse_labels_and_niches.ipynb` | coarse cell types (DCIS_1 and DCIS_2 → DCIS; Prolif_Invasive_Tumor and Invasive_Tumor → Invasive_Tumor), niche composition, KMeans with k = 5 | `predictions_resolvi_proseg_coarse`, `kmeans_alpha_5` |
| 6 | `5_coarse_labels_and_niches.ipynb`, last section (commented out) | CD4 and CD8 T cells relabelled from the two Leiden clusters of a scVIVA embedding of the T cells: 2,668 cells switched between CD4+ and CD8+ T cells, no other change.  | `predictions_resolvi_proseg_coarse_corrected` |
