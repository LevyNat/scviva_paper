# MERFISH mouse brain (Fig. 2, 3, S1A-S5, S10-S14)

Input: `adata_M1_M2_core_6_sections.h5ad` (6 sections of the Zhuang et al. atlas, mice M1 and M2).

**Main benchmark (Fig. 2A-C)**

| step | command | output |
|---|---|---|
| 1. baselines | `(cd baselines && ./run_benchmark.sh brain_merfish 0)` | BANKSY and SimVI embeddings, written into the data file |
| 2. Nicheformer (zero-shot and fine-tuned) | `nicheformer-benchmark/` (own environment): `src/nicheformer/experiments/pretraining_fine_tune.py`, `get_embeddings.py`, then `add_finetuned_embedding.py` | `X_nicheformer`, `X_nicheformer_e5` in the data file |
| 3. scANVI + scVIVA + scIB | `python runner.py merfish_brain/Figure_2` (from the repo root) | scIB tables, bar plots and scatters in `merfish_brain/Figure_2/figures/` |

**Other experiments**, each with its own README:

| figure | folder |
|---|---|
| Fig. 2D-F, S4 (duplicated astrocytes) | `duplicate_region/` |
| Fig. S1A (annotation errors) | `robustness_to_annotation_error/` |
| Fig. S2 (no cell type labels) | `unlabeled/` |
| Fig. S3 (ablations, K and β) | `ablation/` |
| Fig. S5 (synthetic gene) | `synthetic_gene/` |
| Fig. 3, S10-S14 (Hotspot modules of astrocytes) | `hotspot_astro_benchmark.ipynb`, after the main benchmark |
