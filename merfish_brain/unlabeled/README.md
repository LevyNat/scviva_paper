# scVIVA without cell type labels: MERFISH mouse brain (Fig. S2)

scVIVA is trained on Leiden clusters of the scVI embedding instead of the annotated cell types, and compared with
scVIVA trained on the cell types using scIB scores computed with the original labels.

| step | command | output |
|---|---|---|
| 1. choose the Leiden resolution (label-free) | `python select_resolution.py` | `merfish_brain/figures/revisions_benchmark/leiden_resolution_selection_brain.csv`: number of clusters, silhouette (exact, on all cells, computed on the GPU), Calinski-Harabasz and Davies-Bouldin for resolutions 0.05-2.00. The resolution that maximises the silhouette (0.35, 25 clusters) is listed in `FIGURE_LEIDEN_RES` in `params.py`, next to 0.15 (17 clusters, the number of annotated cell types) |
| 2. train and score | `python runner_no_labels.py` | one scVIVA model per resolution in `LEIDEN_RES`; `scib_cell_type_results_ablation_nolabel.csv`, `BioCons_cell_type_ablation_nolabel_res<res>.csv` for each resolution in `FIGURE_LEIDEN_RES`; `merfish_brain/checkpoints/adata_M1_M2_core_6_sections/adata_M1_M2_core_6_sections_nolabel_nicheVI.h5ad` |
| 3. plot | `merfish_brain/Figure_S2_S3.ipynb`, section "Fig. S2" | `X_scVI_umap_nolabel.svg` (A: cell types and the Leiden clusters at each resolution), `nolabels_scib_scatter.svg` (B), `nolabels_region_celltype.svg` (C) |

Step 3 also reads the labeled scVIVA results of `python runner.py merfish_brain/Figure_2`.
The scVI and scVIVA models are saved in `merfish_brain/checkpoints/adata_M1_M2_core_6_sections/`; the runner loads them when they exist.
