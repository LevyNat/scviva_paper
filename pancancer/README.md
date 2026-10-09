# Vizgen pan-cancer atlas (Fig. 5, S18)

MERSCOPE FFPE immuno-oncology release: 8 cancer types, re-segmented with Proseg (10,373,000 cells × 550 genes).

## Figures

| figure | notebook | reads |
|---|---|---|
| Fig. 5A (UMAP by cancer type and cell type), 5B (tissue slices by cell type) | `plot_and_process_dataset.ipynb` | `pancancer_resolvi_nicheVI.h5ad`, `pancancer_resolvi_nicheVI_T_Cell_filtered.h5ad` |
| Fig. S18A-B (T cell filtering: counts, Leiden 0.5, one-vs-all markers) | `T_cell_filtering.ipynb` | `pancancer_resolvi_nicheVI_T_Cell.h5ad` |
| Fig. 5C-H (Hotspot modules, T cell composition per sample, niche composition, DE of cytotoxic T cells and Tregs), S18C | `filtered_T_cell_analysis.ipynb` | `pancancer_resolvi_nicheVI_T_Cell_filtered.h5ad`, `pancancer_resolvi_nicheVI.h5ad`, scVIVA model `nichevae_s10_scanvi_lr0.0005_poisson_1000.pt` (for the DE) |
