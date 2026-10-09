# Robustness to annotation errors: MERFISH mouse brain (Fig. S1A)

Before training, a fraction of the cell type labels (0, 1, 5 and 10%) is switched at random. For every flip rate
and seed (34, 42, 123), scANVI and scVIVA are trained on the noisy labels, then scored with scIB against the
original labels on validation cells.

| step | command | output |
|---|---|---|
| 1. train and score | `python runner.py` (from this folder) | `merfish_brain/figures/robustness_to_annotation_error/seeds/`: `scib_val_results_robustness_seed{34,42,123}.csv` (scIB per flip rate), per-cell-type tables, `losses/` (training histories) |
| 2. plot | `python plot_robustness_evaluation.py` | `seeds/final/`: `scib_robustness_f0f10.svg` (left panel), `loss_robustness_f0f10.svg` (right panel), `overfit_gap_f0f10.svg` |

- Input: `adata_M1_M2_core_6_sections.h5ad` (`params.py`).
- Flip rates and seeds are set in `params.py` (`FLIP_RATES`, `SEEDS_TO_TEST`).
- The 24 models are saved in `merfish_brain/checkpoints/adata_M1_M2_core_6_sections_seed/` (`RUN_NAME` in `params.py`). The runner loads a model when it exists there, so step 1 reproduces the published tables without retraining.
