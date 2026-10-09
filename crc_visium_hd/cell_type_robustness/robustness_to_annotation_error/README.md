# Robustness to annotation errors: Visium HD colorectal cancer (Fig. S1B)

Same experiment as the MERFISH one (`merfish_brain/robustness_to_annotation_error/`): cell type labels are switched
at random before training, and scANVI and scVIVA are scored with scIB against the original labels on validation
cells, for three seeds (34, 42, 123). Fig. S1B shows flip rates 0-10%.

| step | command | output |
|---|---|---|
| 1. train and score | `python runner.py` (from this folder) | `crc_visium_hd/figures/robustness/seeds/`: `scib_val_results_robustness_seed{34,42,123}.csv`, per-cell-type tables, `losses/` |
| 2. plot Fig. S1B | `python plot_robustness_evaluation.py` | `seeds/journal/`: `scib_robustness_nichevi_f0f10.svg`, `scib_robustness_scanvi_f0f10.svg` (left), `loss_robustness_f0f10.svg`, `loss_robustness_scanvi_f0f10.svg` (right), overfitting-gap plots |

Additional scripts (not needed for Fig. S1B):

| script | what it does |
|---|---|
| `run_experiment.sh` | runs step 1, then `aggregate_results.py`; `--ablation` runs the ablated models only (`runner.py --ablation-only`) |
| `aggregate_results.py` | scIB vs flip rate, mean ± std over seeds, for all flip rates in `params.py` (up to 60%) |
| `plot_cell_type_robustness.py` | per-cell-type cell type preservation vs flip rate |
| `plot_losses.py` | training and validation loss curves per flip rate, from the saved models |

- Input: `adata_legacy_hvg4k.h5ad` (`params.py`).
- Flip rates, seeds and the ablated configurations are set in `params.py`.
- The models are saved in `crc_visium_hd/checkpoints/adata_legacy_hvg4k_robustness/`; the runner loads them when they exist.
