# Slide-tags human tonsil (Fig. S9)

| step | command | output | figure |
|---|---|---|---|
| 1. build the data | `slidetags_build.ipynb`: raw counts from the Single Cell Portal files, tissue regions (KMeans, k = 3, on the niche cell-type composition → T zone, germinal center, mantle zone), top 10,000 seurat_v3 HVGs, spatial maps of the cell types and regions | `tonsil_fullgenes.h5ad`, `tonsil_nl_10000hvg.h5ad`, spatial maps | S9A-B |
| 2. benchmark | `tonsil/run_panel.sh 10000hvg` | models, scIB tables, region vs cell type scatter | S9C |
| 3. analysis | `tonsil/tonsil_10000hvg.ipynb` | naive vs memory B cells, plasma cells | S9D-H |

Figure S9 panels, all in `figures/tonsil_10000hvg/` (svg and png):

| panel | file |
|---|---|
| A | `tonsil_spatial_cell_types` |
| B | `tonsil_spatial_regions` |
| C | `region_celltype_scatter_10000hvg_noPCA` |
| D | `tonsil_10000hvg_bcell_confusion_k2` |
| E | `tonsil_10000hvg_bcell_canonical_markers` |
| F | `tonsil_10000hvg_bcell_de_recall_precision` |
| G | `tonsil_10000hvg_plasma_umap_region_baselines` |
| H | `tonsil_10000hvg_plasma_markers_umap` |

`run_panel.sh` runs, in order:
- `tonsil/runner_no_labels.py`: scVI, Leiden pseudo-labels, scVIVA trained on them. The Leiden resolution is chosen without labels on the scVI embedding (silhouette, Calinski-Harabasz and Davies-Bouldin over resolutions 0.05-2.00; majority vote, silhouette breaks ties): 0.15, 7 clusters.
- `baselines/run_benchmark.sh slidetags_tonsil_10000hvg`: BANKSY and SimVI.
- `python runner.py slidetags_tonsil/tonsil_10000hvg`: scANVI and scVIVA.
- `tonsil/score_panel.py 10000hvg`: cell type and region preservation for every method.
- `tonsil/plot_scatter.py 10000hvg --drop PCA`: the scatter of Fig. S9C.

The runners load the provided models when they exist in `checkpoints/`.
