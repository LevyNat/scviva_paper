# Synthetic gene: MERFISH mouse brain (Fig. S5)

A synthetic gene whose expression follows the abundance of a synthetic cell type in the niche is added to the data.
scVIVA is then trained with and without that cell type in the niche composition. 

| step | command | output |
|---|---|---|
| 1. add the synthetic gene and cell type | `synthetic_gene_playground.ipynb` | `adata_M1_M2_core_6_sections_with_synthetic_gene.h5ad` |
| 2. train | `python runner_synthetic_gene.py` (from this folder; `params_positive.py`) | models in `merfish_brain/checkpoints/adata_M1_M2_core_6_sections_with_synthetic_gene/` |
| 3. figures | `synthetic_gene_playground.ipynb` | reconstruction errors with and without the synthetic cell type, Wilcoxon test |

Helper functions are in `synthetic_gene_utils.py`.
