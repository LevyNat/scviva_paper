import argparse
import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import scanpy as sc

from rich import print
import os
import matplotlib.pyplot as plt
import scvi
import nichevi

from params import setup, niche_setup, ablation_setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

# --- Parse command-line arguments ---#
parser = argparse.ArgumentParser()
parser.add_argument("--ablation-only", action="store_true", help="Run ablation experiments only (skip main training loop)")
args = parser.parse_args()

# Note: scvi.settings.seed is now set per-configuration in the training loop
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"


def flip_labels(labels, error_rate, seed=0):
    """
    Flip labels with a specified error rate, ensuring flipped cells get a different label.

    Parameters
    ----------
    labels : array-like
        Original labels
    error_rate : float
        Proportion of labels to flip (between 0 and 1)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    flipped_labels : np.ndarray
        Labels with error_rate proportion flipped to different labels
    flip_idx : np.ndarray
        Indices of flipped cells
    """
    np.random.seed(seed)
    labels = np.array(labels).copy()
    n_flip = int(error_rate * len(labels))

    if n_flip == 0:
        return labels, np.array([])

    # Select cells to flip
    flip_idx = np.random.choice(len(labels), n_flip, replace=False)

    # Get unique labels
    unique_labels = np.unique(labels)

    # If there's only one unique label, cannot flip
    if len(unique_labels) == 1:
        print("Warning: Only one unique label found. Cannot flip labels.")
        return labels, np.array([])

    # Flip each selected cell to a different label
    for i in flip_idx:
        current = labels[i]
        # Create list of all other labels (excluding current)
        other_labels = [lbl for lbl in unique_labels if lbl != current]

        # Randomly select a different label
        new_label = np.random.choice(other_labels)
        labels[i] = new_label

    # Verify that flips actually occurred
    return labels, flip_idx


def save_classification_metrics(model, model_name, error_rate, save_folder, seed):
    """
    Save classification metrics from a trained model to CSV.

    Parameters
    ----------
    model : scvi.model.SCANVI or nichevi.nicheSCVI
        Trained model with classification metrics in history
    model_name : str
        Name of the model (e.g., 'scanvi', 'nichevi_s10')
    error_rate : float
        Error rate used for label flipping
    save_folder : str
        Folder to save the metrics
    seed : int
        Random seed used for training
    """
    # Extract classification metrics from history
    metrics_to_save = [
        "classification_loss",
        "calibration_error",
        "accuracy",
        "f1_score",
    ]

    # Debug: Print all history keys
    print(f"[DEBUG] History keys for {model_name}: {list(model.history.keys())}")

    # Check if metrics exist in history
    available_metrics = {}
    for split in ["train", "validation"]:
        for metric in metrics_to_save:
            key = f"{split}_{metric}"
            if key in model.history:
                # Check if it's a list or scalar
                value = model.history[key]
                if isinstance(value, (list, np.ndarray)):
                    available_metrics[key] = value
                    print(f"[DEBUG] Found {key}: {len(value)} epochs")
                else:
                    # If scalar, it's the final value - skip it
                    print(f"Skipping {key} - scalar value: {value}")

    if not available_metrics:
        print(f"Warning: No classification metrics found in model history for {model_name}")
        print(f"Available history keys: {list(model.history.keys())}")
        return

    # Create DataFrame with epochs as index
    df = pd.DataFrame(available_metrics)
    df.index.name = "epoch"

    # Save to CSV
    filename = f"{model_name}_seed{seed}_flip{error_rate}_classification_metrics.csv"
    save_path = os.path.join(save_folder, filename)
    df.to_csv(save_path)
    print(f"Saved classification metrics to {save_path}")

    # Also save the entire history as pickle for backup
    history_filename = f"{model_name}_seed{seed}_flip{error_rate}_history.pkl"
    history_path = os.path.join(save_folder, history_filename)
    import pickle

    with open(history_path, "wb") as f:
        pickle.dump(dict(model.history), f)
    print(f"Saved full history to {history_path}")


def plot_classification_metrics_comparison(models_dict, error_rates, save_folder, model_type="scanvi"):
    """
    Plot classification metrics for models trained with different error rates.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping error_rate -> model
    error_rates : list
        List of error rates used
    save_folder : str
        Folder to save the plots
    model_type : str
        Type of model ('scanvi' or 'nichevi')
    """
    metrics = ["classification_loss", "calibration_error", "accuracy"]
    n_models = len(error_rates)

    fig, axes = plt.subplots(nrows=3, ncols=n_models, figsize=(3.5 * n_models, 9), sharey="row", sharex=True)

    # Handle case with single model
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    def plot_metric(ax, metric, model, title, lw=1.25):
        train_key = f"train_{metric}"
        val_key = f"validation_{metric}"

        if train_key in model.history:
            ax.plot(
                model.history[train_key],
                label="train",
                color="darkgreen",
                linewidth=lw,
            )
        if val_key in model.history:
            ax.plot(
                model.history[val_key],
                label="validation",
                color="firebrick",
                linewidth=lw,
            )
        ax.legend()
        ax.set_title(title)
        ax.grid(alpha=0.3)

    metric_labels = ["Classification loss", "Calibration error", "Accuracy"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for j, error_rate in enumerate(error_rates):
            if error_rate in models_dict:
                model = models_dict[error_rate]
                title = f"Error rate: {error_rate}"
                plot_metric(axes[i, j], metric, model, title)

        # Add ylabel to leftmost plot
        axes[i, 0].set_ylabel(label)

    # Add xlabel to bottom plots
    for j in range(n_models):
        axes[-1, j].set_xlabel("Epoch")

    fig.suptitle(f"{model_type.upper()} Classification Metrics", fontsize=14, y=1.0)
    fig.tight_layout()

    # Save figure
    filename = f"{model_type}_classification_metrics_comparison.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight", dpi=300)
    print(f"Saved classification metrics plot to {save_path}")


# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

# path_to_save = os.path.join("../checkpoints", data_file_name)
path_to_save = "/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/adata_legacy_hvg4k_robustness"
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))


# adata.layers["counts"] = adata.raw.X.copy()
print(adata)
# Filter out cells with very low counts to prevent NaN during training
counts = adata.layers["counts"]
if hasattr(counts, "toarray"):
    cell_counts = counts.toarray().sum(axis=1)
else:
    cell_counts = counts.sum(axis=1)

n_cells_before = adata.n_obs
min_counts = 10  # Minimum counts per cell
adata = adata[cell_counts >= min_counts].copy()
print(f"Filtered {n_cells_before - adata.n_obs} cells with < {min_counts} counts")
print(adata)
# print(
#     "Will save the adata as: ",
#     os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad"),
# )


# ----------scVI----------#


scvi_is_trained = False
nichevi_is_trained = False
save_scvi = True
compute_umap = True
compute_scib = True
scib_proportion = 0.3

PALETTE = "tab20"

print(
    f"{setup.EXPRESSION_MODEL} trained: ",
    scvi_is_trained,
    "nicheVI trained: ",
    nichevi_is_trained,
    "compute UMAP: ",
    compute_umap,
)

history_setup = {}
latent_keys_niche = []
latent_key_scvi = []

# For storing models for classification metrics plotting
scanvi_models = {}
nichevi_models = {}

# Create folder for classification metrics
losses_folder = os.path.join(setup.FIGURES_FOLDER, "losses")
os.makedirs(losses_folder, exist_ok=True)

# For storing and verifying train/val indices across flip rates (per seed)
# Keys are seeds, values are (train_indices, val_indices)
reference_indices_by_seed = {}

if args.ablation_only:
    print("[bold yellow]===== ABLATION ONLY MODE - Skipping main training loop =====[/bold yellow]")
    niche_setup = {}  # Skip main loop
    compute_scib = False  # Skip main benchmarks (ablation has its own)

for setting in niche_setup.keys():
    print("[bold green]" + setting + "[/bold green]")
    setup_dict = niche_setup[setting]

    # Get the seed for this configuration
    config_seed = setup_dict["seed"]
    print(f"Using seed: {config_seed}")

    # Set the global seed for this configuration
    scvi.settings.seed = config_seed

    label_to_flip = setup_dict["label_to_flip"]
    print("Prop. of labels to flip: ", label_to_flip)

    # --- Flip and store ---
    original_labels = adata.obs[setup.CELL_TYPE].values

    flipped_labels, flip_idx = flip_labels(original_labels, label_to_flip, seed=config_seed)
    adata.obs["flipped_cell_type"] = flipped_labels
    adata.obs[f"flipped_cell_type_{label_to_flip}"] = flipped_labels

    if scvi_is_trained is False:
        # Check if model already exists at the save path
        filename = f"{setup.EXPRESSION_MODEL}vae_E{setup.N_EPOCHS_SCVI}_{setup.LIKELIHOOD}_seed{config_seed}_flip{label_to_flip}.pt"
        save_path = os.path.join(path_to_save, filename)

        if os.path.exists(save_path):
            print(f"[bold yellow]Model already exists at {save_path}, loading instead of training...[/bold yellow]")
            if setup.EXPRESSION_MODEL == "scanvi":
                scvivae = scvi.model.SCANVI.load(
                    dir_path=save_path,
                    adata=adata,
                )
            elif setup.EXPRESSION_MODEL == "scvi":
                scvivae = scvi.model.SCVI.load(
                    dir_path=save_path,
                    adata=adata,
                )
        else:
            # Model doesn't exist, train it
            if setup.EXPRESSION_MODEL == "scanvi":
                scvi.model.SCANVI.setup_anndata(
                    adata,
                    layer="counts",
                    unlabeled_category="ignore",
                    batch_key=setup.BATCH,
                    labels_key="flipped_cell_type",
                )

                scvivae = scvi.model.SCANVI(
                    adata,
                    gene_likelihood=setup.LIKELIHOOD,
                    n_layers=setup.N_LAYERS,
                    n_latent=setup.N_LATENT,
                    linear_classifier=True,  # seems to improve scib?
                )

            if setup.EXPRESSION_MODEL == "scvi":
                scvi.model.SCVI.setup_anndata(
                    adata,
                    layer="counts",
                    batch_key=setup.BATCH,
                )

                scvivae = scvi.model.SCVI(
                    adata,
                    gene_likelihood=setup.LIKELIHOOD,
                    n_layers=setup.N_LAYERS,
                    n_latent=setup.N_LATENT,
                )

            scvivae.train(
                max_epochs=setup.N_EPOCHS_SCVI,
                train_size=0.8,
                validation_size=0.2,
                batch_size=setup.BATCH_SIZE_SCVI,
                plan_kwargs=dict(
                    lr=setup.LR_SCVI,
                    n_epochs_kl_warmup=setup.KL_WARMUP,
                    weight_decay=setup.WEIGHT_DECAY,
                    optimizer=setup.OPTIMIZER,
                ),
                # trainer_kwargs=dict(check_val_every_n_epoch=1),
                early_stopping=True,
            )

            if save_scvi:
                # scvivae.save(
                #     dir_path=path_to_save
                #     + f"/{setup.EXPRESSION_MODEL}vae_E"
                #     + str(setup.N_EPOCHS_SCVI)
                #     + "_"
                #     + str(setup.LIKELIHOOD)
                #     # + "L"
                #     + ".pt",
                #     save_anndata=False,
                # )
                scvivae.save(save_path, save_anndata=False, overwrite=True)

    if scvi_is_trained:
        filename = f"{setup.EXPRESSION_MODEL}vae_E{setup.N_EPOCHS_SCVI}_{setup.LIKELIHOOD}_seed{config_seed}_flip{label_to_flip}.pt"
        load_path = os.path.join(path_to_save, filename)

        if setup.EXPRESSION_MODEL == "scanvi":
            scvivae = scvi.model.SCANVI.load(
                dir_path=load_path,
                adata=adata,
            )
        if setup.EXPRESSION_MODEL == "scvi":
            scvivae = scvi.model.SCVI.load(
                dir_path=load_path,
                adata=adata,
            )

    latent_key = "X_scVI" if setup.EXPRESSION_MODEL == "scvi" else "X_scanvi"

    latent_key += f"_seed{config_seed}_flip{label_to_flip}"

    adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)

    latent_key_scvi.append(latent_key)

    # Store and verify train/val indices (per seed)
    if config_seed not in reference_indices_by_seed:
        reference_indices_by_seed[config_seed] = (scvivae.train_indices, scvivae.validation_indices)
        print(
            f"[bold blue]Stored reference indices for seed {config_seed} - Train: {len(scvivae.train_indices)}, Val: {len(scvivae.validation_indices)}[/bold blue]"
        )
    else:
        # Verify consistency across flip rates for the same seed
        ref_train, ref_val = reference_indices_by_seed[config_seed]
        assert np.array_equal(
            scvivae.train_indices, ref_train
        ), f"scANVI train indices differ at flip={label_to_flip} seed={config_seed}!"
        assert np.array_equal(
            scvivae.validation_indices, ref_val
        ), f"scANVI val indices differ at flip={label_to_flip} seed={config_seed}!"
        print(f"[green]✓ Verified scANVI indices match at flip={label_to_flip} seed={config_seed}[/green]")

    if compute_umap:
        sc.pp.neighbors(adata, use_rep=latent_key)
        sc.tl.umap(adata, min_dist=0.3)

        sc.pl.umap(
            adata,
            color=[setup.CELL_TYPE, f"flipped_cell_type_{label_to_flip}"],
            frameon=False,
            ncols=1,
            # palette=cc.glasbey_dark,
            palette=PALETTE,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{latent_key}_seed{config_seed}_flip{label_to_flip}_umap.png",
            bbox_inches="tight",
            dpi=500,
        )
        plt.savefig(f"{setup.FIGURES_FOLDER}{latent_key}_umap.svg", bbox_inches="tight", dpi=500)

    history_setup[latent_key] = scvivae.history

    print(scvivae.history.keys())

    # Save classification metrics for scANVI/scVI
    if setup.EXPRESSION_MODEL == "scanvi":
        save_classification_metrics(
            scvivae,
            model_name=setup.EXPRESSION_MODEL,
            error_rate=label_to_flip,
            save_folder=losses_folder,
            seed=config_seed,
        )
        scanvi_models[label_to_flip] = scvivae

    print(f"{setup.EXPRESSION_MODEL} done...")

    # ----------NicheVI----------#

    print("[bold green]" + setting + "[/bold green]")
    setup_dict = niche_setup[setting]

    # preprocessing function to populate adata.obsm with the keys 'neighborhood_composition',
    # 'qz1_m', 'qz1_var', 'niche_indexes', 'niche_distances', 'qz1_m_niche_knn', 'qz1_var_niche_knn', 'qz1_m_niche_ct',
    # 'qz1_var_niche_ct'
    filename_nichevae = f"nichevae_{setting}_E{setup.N_EPOCHS_NICHEVI}_" f"seed{config_seed}_flip{label_to_flip}.pt"
    path_to_save_nichevae = os.path.join(path_to_save, filename_nichevae)

    # path_to_save_nichevae = path_to_save + "/nichevae_" + setting + "_" + str(setup.N_EPOCHS_NICHEVI) + ".pt"

    if setup_dict["niche_expression"] == "pca":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm["X_pca"]

    if setup_dict["niche_expression"] == "scvi":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm[latent_key]

    if setup_dict["niche_expression"] == "resolvi":
        NICHE_LIKELIHOOD = "gaussian"
        # adata_resolvi = ad.read_h5ad(
        #     os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad")
        # )
        adata.obsm["qz1_m"] = adata.obsm["X_resolvi"].copy()

    if setup_dict["niche_expression"] == "gaussian_counts":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.X.toarray()

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": "flipped_cell_type",
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(
        adata,
        k_nn=setup.K_NN,
        **setup_kwargs,
    )

    nichevi.nicheSCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=setup.BATCH,
        **setup_kwargs,
    )

    if nichevi_is_trained is False:
        # check if path_to_save_nichevae exists
        if os.path.exists(path_to_save_nichevae):
            print("Model already exists...")
            # get out of the loop
            # continue

        nichevae = nichevi.nicheSCVI(
            adata,
            cell_rec_weight=setup_dict["cell_rec_weight"],
            latent_kl_weight=setup_dict["latent_kl_weight"],
            spatial_weight=setup_dict["spatial_weight"],
            niche_rec_weight=setup_dict["niche_rec_weight"],
            compo_rec_weight=setup_dict["compo_rec_weight"],
            niche_likelihood=NICHE_LIKELIHOOD,
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
            # n_heads=setup_dict["n_heads"],
            # n_tokens_decoder=setup_dict["n_tokens_decoder"],
            n_layers_niche=setup_dict["n_layers_niche"],
            n_layers_compo=setup_dict["n_layers_compo"],
            n_hidden_niche=setup_dict["n_hidden_niche"],
            n_hidden_compo=setup_dict["n_hidden_compo"],
            n_latent=setup_dict["n_latent"],
            use_batch_norm="both" if setup.USE_BATCH_NORM else "none",
            use_layer_norm="none" if setup.USE_BATCH_NORM else "both",
            prior_mixture=setup_dict["prior_mixture"],
            semisupervised=True,
            linear_classifier=True,
            # prior_mixture_k=setup_dict["prior_mixture_k"],
        )

        nichevae.train(
            max_epochs=setup.N_EPOCHS_NICHEVI,
            train_size=0.8,
            validation_size=0.2,
            early_stopping=True,
            check_val_every_n_epoch=1,
            batch_size=setup.BATCH_SIZE_NICHEVI,
            plan_kwargs=dict(
                lr=setup.LR_NICHEVI,
                # n_epochs_kl_warmup=setup.KL_WARMUP,
                n_epochs_kl_warmup=setup_dict["kl_warmup"],
                # n_steps_kl_warmup=setup.N_STEPS_KL_WARMUP,
                # max_kl_weight=setup.MAX_KL_WEIGHT,
                max_kl_weight=setup_dict["max_kl_weight"],
                n_epochs_spatial_warmup=setup.SPATIAL_WARMUP,
                min_spatial_weight=setup.MIN_SPATIAL_WEIGHT,
                max_spatial_weight=setup.MAX_SPATIAL_WEIGHT,
                optimizer=setup.OPTIMIZER,
                weight_decay=setup.WEIGHT_DECAY,
                reduce_lr_on_plateau=setup.REDUCE_LR_ON_PLATEAU,
            ),
            early_stopping_patience=100,  # trick because sometimes KL goes up.
            # lr_scheduler_metric="reconstruction_loss_validation",
        )

        nichevae.save(
            dir_path=path_to_save_nichevae,
            save_anndata=False,
            overwrite=True,
        )

    if nichevi_is_trained:
        nichevae = nichevi.nicheSCVI.load(
            dir_path=path_to_save_nichevae,
            adata=adata,
        )

    history_setup[setting] = nichevae.history
    print(nichevae.history.keys())

    # Save classification metrics for NicheVI
    save_classification_metrics(
        nichevae,
        model_name=f"nichevi_{setting}",
        error_rate=label_to_flip,
        save_folder=losses_folder,
        seed=config_seed,
    )
    nichevi_models[label_to_flip] = nichevae

    latent_key_niche = setting + "_X_nicheVI" + f"_seed{config_seed}_flip{label_to_flip}"
    latent_keys_niche.append(latent_key_niche)
    adata.obsm[latent_key_niche] = nichevae.get_latent_representation(batch_size=1024)

    # Verify NicheVI indices match scANVI indices for the same seed
    ref_train, ref_val = reference_indices_by_seed[config_seed]
    assert np.array_equal(
        nichevae.train_indices, ref_train
    ), f"NicheVI train indices differ from scANVI at flip={label_to_flip} seed={config_seed}!"
    assert np.array_equal(
        nichevae.validation_indices, ref_val
    ), f"NicheVI val indices differ from scANVI at flip={label_to_flip} seed={config_seed}!"
    print(f"[green]✓ Verified NicheVI indices match at flip={label_to_flip} seed={config_seed}[/green]")

    # adata.obs[setting + "_compo_error"] = nichevae.get_composition_error(return_mean=False).cpu()
    # adata.obs[setting + "_niche_error"] = nichevae.get_niche_error(return_mean=False).cpu()

    if compute_umap:
        sc.pp.neighbors(adata, use_rep=latent_key_niche)
        sc.tl.umap(adata, min_dist=0.3)

        sc.pl.umap(
            adata,
            color=[setup.CELL_TYPE, f"flipped_cell_type_{label_to_flip}"],
            frameon=False,
            ncols=1,
            # palette=cc.glasbey_dark,
            palette=PALETTE,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{setting}_seed{config_seed}_flip{label_to_flip}_umap.png",
            bbox_inches="tight",
            dpi=500,
        )
        plt.savefig(
            f"{setup.FIGURES_FOLDER}{setting}_seed{config_seed}_flip{label_to_flip}_umap.svg",
            bbox_inches="tight",
            dpi=500,
        )

# --- Plot classification metrics comparison ---#

if scanvi_models and setup.EXPRESSION_MODEL == "scanvi":
    error_rates = sorted(scanvi_models.keys())
    plot_classification_metrics_comparison(
        scanvi_models,
        error_rates,
        losses_folder,
        model_type="scanvi",
    )

if nichevi_models:
    error_rates = sorted(nichevi_models.keys())
    plot_classification_metrics_comparison(
        nichevi_models,
        error_rates,
        losses_folder,
        model_type="nichevi",
    )

# ---------- Ablation Experiments (eta0, s0) ----------#
# These reuse the scANVI model from the main loop and only train NicheVI with different weights.
# Only fixed-sample benchmarking is run for ablation configs.

latent_keys_ablation = []

for setting in ablation_setup.keys():
    print("[bold green]ABLATION: " + setting + "[/bold green]")
    setup_dict = ablation_setup[setting]

    config_seed = setup_dict["seed"]
    scvi.settings.seed = config_seed

    label_to_flip = setup_dict["label_to_flip"]
    print(f"Using seed: {config_seed}, flip rate: {label_to_flip}")

    # Flip labels (same logic as main loop)
    original_labels = adata.obs[setup.CELL_TYPE].values
    flipped_labels, flip_idx = flip_labels(original_labels, label_to_flip, seed=config_seed)
    adata.obs["flipped_cell_type"] = flipped_labels

    # Reuse existing scANVI embedding
    latent_key = "X_scanvi" if setup.EXPRESSION_MODEL == "scanvi" else "X_scVI"
    latent_key += f"_seed{config_seed}_flip{label_to_flip}"

    if latent_key not in adata.obsm:
        # Need to load scANVI model to get embedding
        filename = f"{setup.EXPRESSION_MODEL}vae_E{setup.N_EPOCHS_SCVI}_{setup.LIKELIHOOD}_seed{config_seed}_flip{label_to_flip}.pt"
        load_path = os.path.join(path_to_save, filename)
        if setup.EXPRESSION_MODEL == "scanvi":
            scvi.model.SCANVI.setup_anndata(
                adata, layer="counts", unlabeled_category="ignore",
                batch_key=setup.BATCH, labels_key="flipped_cell_type",
            )
            scvivae = scvi.model.SCANVI.load(dir_path=load_path, adata=adata)
        else:
            scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=setup.BATCH)
            scvivae = scvi.model.SCVI.load(dir_path=load_path, adata=adata)
        adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)

    # NicheVI preprocessing using the scANVI embedding
    NICHE_LIKELIHOOD = "gaussian"
    adata.obsm["qz1_m"] = adata.obsm[latent_key]

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": "flipped_cell_type",
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(adata, k_nn=setup.K_NN, **setup_kwargs)
    nichevi.nicheSCVI.setup_anndata(adata, layer="counts", batch_key=setup.BATCH, **setup_kwargs)

    filename_nichevae = f"nichevae_{setting}_E{setup.N_EPOCHS_NICHEVI}_seed{config_seed}_flip{label_to_flip}.pt"
    path_to_save_nichevae = os.path.join(path_to_save, filename_nichevae)

    if nichevi_is_trained is False:
        if os.path.exists(path_to_save_nichevae):
            print("Ablation model already exists...")

        nichevae = nichevi.nicheSCVI(
            adata,
            cell_rec_weight=setup_dict["cell_rec_weight"],
            latent_kl_weight=setup_dict["latent_kl_weight"],
            spatial_weight=setup_dict["spatial_weight"],
            niche_rec_weight=setup_dict["niche_rec_weight"],
            compo_rec_weight=setup_dict["compo_rec_weight"],
            niche_likelihood=NICHE_LIKELIHOOD,
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
            n_layers_niche=setup_dict["n_layers_niche"],
            n_layers_compo=setup_dict["n_layers_compo"],
            n_hidden_niche=setup_dict["n_hidden_niche"],
            n_hidden_compo=setup_dict["n_hidden_compo"],
            n_latent=setup_dict["n_latent"],
            use_batch_norm="both" if setup.USE_BATCH_NORM else "none",
            use_layer_norm="none" if setup.USE_BATCH_NORM else "both",
            prior_mixture=setup_dict["prior_mixture"],
            semisupervised=True,
            linear_classifier=True,
        )

        nichevae.train(
            max_epochs=setup.N_EPOCHS_NICHEVI,
            train_size=0.8,
            validation_size=0.2,
            early_stopping=True,
            check_val_every_n_epoch=1,
            batch_size=setup.BATCH_SIZE_NICHEVI,
            plan_kwargs=dict(
                lr=setup.LR_NICHEVI,
                n_epochs_kl_warmup=setup_dict["kl_warmup"],
                max_kl_weight=setup_dict["max_kl_weight"],
                n_epochs_spatial_warmup=setup.SPATIAL_WARMUP,
                min_spatial_weight=setup.MIN_SPATIAL_WEIGHT,
                max_spatial_weight=setup.MAX_SPATIAL_WEIGHT,
                optimizer=setup.OPTIMIZER,
                weight_decay=setup.WEIGHT_DECAY,
                reduce_lr_on_plateau=setup.REDUCE_LR_ON_PLATEAU,
            ),
            early_stopping_patience=100,
        )

        nichevae.save(dir_path=path_to_save_nichevae, save_anndata=False, overwrite=True)

    if nichevi_is_trained:
        nichevae = nichevi.nicheSCVI.load(dir_path=path_to_save_nichevae, adata=adata)

    latent_key_ablation = setting + "_X_nicheVI" + f"_seed{config_seed}_flip{label_to_flip}"
    latent_keys_ablation.append(latent_key_ablation)
    adata.obsm[latent_key_ablation] = nichevae.get_latent_representation(batch_size=1024)

    history_setup[setting] = nichevae.history
    print(f"Ablation {setting} done.")

print(f"\n[bold magenta]Ablation training complete. {len(latent_keys_ablation)} embeddings.[/bold magenta]\n")

# --- Save history + latent space for benchmarking ---#

adata.layers["counts"] = csr_matrix(adata.layers["counts"])

print(adata)
print("Saving adata...")

# adata.write_h5ad(
#     os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad"),
# )

adata.write_h5ad(os.path.join(data_dir, data_file))

# # Save the dictionary to a PKL file
# with open(path_to_save + "/models_history.pkl", "wb") as pickle_file:
#     pd.to_pickle(history_setup, pickle_file)


# from plot_history import plot_history

# plot_history()


# ----------Benchmarks----------#

if compute_scib:
    import warnings

    warnings.filterwarnings("ignore")

    batchcorr = BatchCorrection(
        # silhouette_batch=True,
        bras=False,
        ilisi_knn=True,
        kbet_per_label=False,
        graph_connectivity=False,
        pcr_comparison=False,
    )

    biocons = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )

    # from harmony import harmonize

    # adata.obsm["X_banksy_02_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.2"], adata.obs, batch_key=setup.BATCH
    # )

    embedding_obsm_keys = (
        # [latent_key]
        # [] + [setting + "_X_nicheVI" for setting in niche_setup.keys()]
        # + [
        #     # "simvi_both_08",
        #     "banksy_pc_20_lambda_0.2",
        #     # "simvi_interact_08",
        #     # "simvi_intrinsic_08",
        #     "X_banksy_02_harmony",
        #     # "X_resolvi",
        #     #  "X_banksy_04_harmony",
        #     "X_nicheformer",
        #     "X_nicheformer_e5",
        #     "simvi25_intrinsic_mae_50",
        #     # "simvi25_interact_mae_50",
        #     "simvi25_both_mae_50",
        #     "simvi25_all_mae_50",
        # ]
        latent_keys_niche
    )

    if scib_proportion < 1.0:
        adata_sampled = sc.pp.sample(adata, fraction=scib_proportion, copy=True, replace=False, rng=34)
    else:
        adata_sampled = adata.copy()

    # Benchmarking for scVIVA
    bm = Benchmarker(
        adata_sampled,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_robustness.png", bbox_inches="tight", dpi=800)
    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_robustness.svg", bbox_inches="tight", dpi=800)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{setup.FIGURES_FOLDER}scib_cell_type_results_robustness.csv")

    # Benchmarking for scANVI
    bm = Benchmarker(
        adata_sampled,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,
        embedding_obsm_keys=latent_key_scvi,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_robustness_scanvi.png", bbox_inches="tight", dpi=800)
    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_robustness_scanvi.svg", bbox_inches="tight", dpi=800)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{setup.FIGURES_FOLDER}scib_cell_type_results_robustness_scanvi.csv")

    # ===== Train/Val Split Analysis (Per Seed) =====
    # Benchmark on train and validation subsets separately using TRUE labels
    # This shows that embeddings preserve biological structure despite label corruption
    # Run separately for each seed to enable proper comparison

    print("\n[bold magenta]===== Running scIB on Train/Val Splits with True Labels (Per Seed) =====[/bold magenta]\n")

    for bench_seed, (train_indices, val_indices) in reference_indices_by_seed.items():
        print(f"\n[bold yellow]===== Benchmarking for seed {bench_seed} =====[/bold yellow]\n")
        print(f"Train set size: {len(train_indices)}, Validation set size: {len(val_indices)}")

        # Filter embeddings to only those from this seed
        seed_niche_keys = [k for k in latent_keys_niche if f"_seed{bench_seed}_" in k]
        seed_scvi_keys = [k for k in latent_key_scvi if f"_seed{bench_seed}_" in k]

        print(f"NicheVI embeddings for seed {bench_seed}: {len(seed_niche_keys)}")
        print(f"scANVI embeddings for seed {bench_seed}: {len(seed_scvi_keys)}")

        if not seed_niche_keys and not seed_scvi_keys:
            print(f"[yellow]No embeddings found for seed {bench_seed}, skipping...[/yellow]")
            continue

        # Subset to train and validation
        adata_train_full = adata[train_indices].copy()
        adata_val = adata[val_indices].copy()  # Keep full validation set

        # Subsample train set only (val is already small)
        if scib_proportion < 1.0:
            adata_train = sc.pp.sample(adata_train_full, fraction=scib_proportion, copy=True, replace=False, rng=bench_seed)
            print(f"Subsampled train to {adata_train.n_obs} cells (proportion={scib_proportion})")
            print(f"Using full validation set: {adata_val.n_obs} cells")
        else:
            adata_train = adata_train_full

        # --- scIB on Training Set (NicheVI embeddings) ---
        if seed_niche_keys:
            print(f"\n[bold cyan]Benchmarking NicheVI on TRAIN set (seed {bench_seed})...[/bold cyan]")
            bm_train = Benchmarker(
                adata_train,
                batch_key=setup.BATCH,
                label_key=setup.CELL_TYPE,  # Use TRUE labels!
                embedding_obsm_keys=seed_niche_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm_train.benchmark()
            bm_train.plot_results_table(min_max_scale=False, show=False)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_train_robustness_seed{bench_seed}.png", bbox_inches="tight", dpi=800)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_train_robustness_seed{bench_seed}.svg", bbox_inches="tight", dpi=800)
            scib_train_df = bm_train.get_results(min_max_scale=False, clean_names=True)
            scib_train_df.to_csv(f"{setup.FIGURES_FOLDER}scib_train_results_robustness_seed{bench_seed}.csv")
            print(f"[green]✓ Saved train scIB results for seed {bench_seed}[/green]")

            # --- scIB on Validation Set (NicheVI embeddings) ---
            print(f"\n[bold cyan]Benchmarking NicheVI on VALIDATION set (seed {bench_seed})...[/bold cyan]")
            bm_val = Benchmarker(
                adata_val,
                batch_key=setup.BATCH,
                label_key=setup.CELL_TYPE,  # Use TRUE labels!
                embedding_obsm_keys=seed_niche_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm_val.benchmark()
            bm_val.plot_results_table(min_max_scale=False, show=False)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_val_robustness_seed{bench_seed}.png", bbox_inches="tight", dpi=800)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_val_robustness_seed{bench_seed}.svg", bbox_inches="tight", dpi=800)
            scib_val_df = bm_val.get_results(min_max_scale=False, clean_names=True)
            scib_val_df.to_csv(f"{setup.FIGURES_FOLDER}scib_val_results_robustness_seed{bench_seed}.csv")
            print(f"[green]✓ Saved validation scIB results for seed {bench_seed}[/green]")

        # --- scIB on Training Set (scANVI embeddings only) ---
        if seed_scvi_keys:
            print(f"\n[bold cyan]Benchmarking scANVI on TRAIN set (seed {bench_seed})...[/bold cyan]")
            bm_train_scanvi = Benchmarker(
                adata_train,
                batch_key=setup.BATCH,
                label_key=setup.CELL_TYPE,  # Use TRUE labels!
                embedding_obsm_keys=seed_scvi_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm_train_scanvi.benchmark()
            bm_train_scanvi.plot_results_table(min_max_scale=False, show=False)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_train_robustness_scanvi_seed{bench_seed}.png", bbox_inches="tight", dpi=800)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_train_robustness_scanvi_seed{bench_seed}.svg", bbox_inches="tight", dpi=800)
            scib_train_scanvi_df = bm_train_scanvi.get_results(min_max_scale=False, clean_names=True)
            scib_train_scanvi_df.to_csv(f"{setup.FIGURES_FOLDER}scib_train_results_robustness_scanvi_seed{bench_seed}.csv")
            print(f"[green]✓ Saved train scANVI scIB results for seed {bench_seed}[/green]")

            # --- scIB on Validation Set (scANVI embeddings only) ---
            print(f"\n[bold cyan]Benchmarking scANVI on VALIDATION set (seed {bench_seed})...[/bold cyan]")
            bm_val_scanvi = Benchmarker(
                adata_val,
                batch_key=setup.BATCH,
                label_key=setup.CELL_TYPE,  # Use TRUE labels!
                embedding_obsm_keys=seed_scvi_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm_val_scanvi.benchmark()
            bm_val_scanvi.plot_results_table(min_max_scale=False, show=False)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_val_robustness_scanvi_seed{bench_seed}.png", bbox_inches="tight", dpi=800)
            plt.savefig(f"{setup.FIGURES_FOLDER}scib_val_robustness_scanvi_seed{bench_seed}.svg", bbox_inches="tight", dpi=800)
            scib_val_scanvi_df = bm_val_scanvi.get_results(min_max_scale=False, clean_names=True)
            scib_val_scanvi_df.to_csv(f"{setup.FIGURES_FOLDER}scib_val_results_robustness_scanvi_seed{bench_seed}.csv")
            print(f"[green]✓ Saved validation scANVI scIB results for seed {bench_seed}[/green]")

    print("\n[bold magenta]===== Train/Val Split Analysis Complete (All Seeds) =====[/bold magenta]\n")

    # ===== Fixed Sample Benchmarking (Same cells for all seeds) =====
    # Sample once from full dataset with fixed seed, evaluate all embeddings
    # This enables fair cross-seed comparison on identical cells
    print("\n[bold magenta]===== Fixed Sample Benchmarking (Cross-Seed Comparison) =====[/bold magenta]\n")

    FIXED_SAMPLE_SEED = 34  # Fixed seed for sampling - same cells for all embeddings

    if scib_proportion < 1.0:
        adata_fixed_sample = sc.pp.sample(
            adata, fraction=scib_proportion, copy=True, replace=False, rng=FIXED_SAMPLE_SEED
        )
        print(f"Fixed sample: {adata_fixed_sample.n_obs} cells (sampled with seed {FIXED_SAMPLE_SEED})")
    else:
        adata_fixed_sample = adata.copy()
        print(f"Using full dataset: {adata_fixed_sample.n_obs} cells")

    # Benchmark ALL NicheVI embeddings on fixed sample
    if latent_keys_niche:
        print("\n[bold cyan]Benchmarking ALL NicheVI embeddings on fixed sample...[/bold cyan]")
        bm_fixed_niche = Benchmarker(
            adata_fixed_sample,
            batch_key=setup.BATCH,
            label_key=setup.CELL_TYPE,
            embedding_obsm_keys=latent_keys_niche,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batchcorr,
            n_jobs=-1,
        )
        bm_fixed_niche.benchmark()
        bm_fixed_niche.plot_results_table(min_max_scale=False, show=False)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_nichevi.png", bbox_inches="tight", dpi=800)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_nichevi.svg", bbox_inches="tight", dpi=800)
        scib_fixed_niche_df = bm_fixed_niche.get_results(min_max_scale=False, clean_names=True)
        scib_fixed_niche_df.to_csv(f"{setup.FIGURES_FOLDER}scib_fixed_sample_results_nichevi.csv")
        print("[green]✓ Saved fixed sample NicheVI results[/green]")

    # Benchmark ALL scANVI embeddings on fixed sample
    if latent_key_scvi:
        print("\n[bold cyan]Benchmarking ALL scANVI embeddings on fixed sample...[/bold cyan]")
        bm_fixed_scanvi = Benchmarker(
            adata_fixed_sample,
            batch_key=setup.BATCH,
            label_key=setup.CELL_TYPE,
            embedding_obsm_keys=latent_key_scvi,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batchcorr,
            n_jobs=-1,
        )
        bm_fixed_scanvi.benchmark()
        bm_fixed_scanvi.plot_results_table(min_max_scale=False, show=False)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_scanvi.png", bbox_inches="tight", dpi=800)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_scanvi.svg", bbox_inches="tight", dpi=800)
        scib_fixed_scanvi_df = bm_fixed_scanvi.get_results(min_max_scale=False, clean_names=True)
        scib_fixed_scanvi_df.to_csv(f"{setup.FIGURES_FOLDER}scib_fixed_sample_results_scanvi.csv")
        print("[green]✓ Saved fixed sample scANVI results[/green]")

    # Benchmark ablation embeddings on fixed sample
    if latent_keys_ablation:
        print("\n[bold cyan]Benchmarking ABLATION NicheVI embeddings on fixed sample...[/bold cyan]")
        bm_fixed_ablation = Benchmarker(
            adata_fixed_sample,
            batch_key=setup.BATCH,
            label_key=setup.CELL_TYPE,
            embedding_obsm_keys=latent_keys_ablation,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batchcorr,
            n_jobs=-1,
        )
        bm_fixed_ablation.benchmark()
        bm_fixed_ablation.plot_results_table(min_max_scale=False, show=False)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_ablation.png", bbox_inches="tight", dpi=800)
        plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_ablation.svg", bbox_inches="tight", dpi=800)
        scib_fixed_ablation_df = bm_fixed_ablation.get_results(min_max_scale=False, clean_names=True)
        scib_fixed_ablation_df.to_csv(f"{setup.FIGURES_FOLDER}scib_fixed_sample_results_ablation.csv")
        print("[green]✓ Saved fixed sample ablation results[/green]")

    print("\n[bold magenta]===== Fixed Sample Benchmarking Complete =====[/bold magenta]\n")

    # effect on region labels (only on validation set) - Per Seed
    print("\n[bold magenta]===== Cell Type Niche Analysis on Validation Set (Per Seed) =====[/bold magenta]\n")

    for bench_seed, (train_indices, val_indices) in reference_indices_by_seed.items():
        print(f"\n[bold yellow]===== Cell Type Niche Analysis for seed {bench_seed} =====[/bold yellow]\n")

        # Filter embeddings to only those from this seed
        seed_niche_keys = [k for k in latent_keys_niche if f"_seed{bench_seed}_" in k]

        if not seed_niche_keys:
            print(f"[yellow]No NicheVI embeddings found for seed {bench_seed}, skipping...[/yellow]")
            continue

        # Get validation set for this seed
        adata_val_seed = adata[val_indices].copy()

        TRESHOLD = 200
        adata_val_seed.obs["cell_type_niche"] = (
            adata_val_seed.obs[setup.CELL_TYPE].astype(str) + "_" + adata_val_seed.obs[setup.NICHE].astype(str)
        )
        value_counts = adata_val_seed.obs["cell_type_niche"].value_counts()
        cell_types_to_keep = value_counts[value_counts >= TRESHOLD].index
        adata_subset = adata_val_seed[adata_val_seed.obs["cell_type_niche"].isin(cell_types_to_keep)].copy()
        cell_types = adata_subset.obs[setup.CELL_TYPE].unique()
        print(f"Using validation set only with threshold={TRESHOLD}")
        print(f"Found {len(cell_types_to_keep)} cell_type_niche combinations above threshold")

        # Initialize dictionaries
        dict_DE = {"cell_type": []}
        dict_BC = {"cell_type": []}

        # Add embedding keys to each dictionary
        for embedding in seed_niche_keys:
            dict_BC[embedding] = []
            dict_DE[embedding] = []

        # Loop through each cell type
        for cell_type in adata_subset.obs[setup.CELL_TYPE].unique().tolist():
            adata_subset_type = adata_subset[adata_subset.obs[setup.CELL_TYPE] == cell_type].copy()
            type_niche_list = adata_subset_type.obs["cell_type_niche"].unique().tolist()

            # Skip cell types with only one niche
            if len(type_niche_list) == 1:
                print(f"Skipping {cell_type} as it only has one niche")
                continue

            print(cell_type)

            # Append cell type to dictionaries
            for d in [
                dict_DE,
                dict_BC,
            ]:
                d["cell_type"].append(cell_type)

            # Create and run the benchmarker
            bm = Benchmarker(
                adata_subset_type,
                batch_key=setup.BATCH,
                label_key="cell_type_niche",
                embedding_obsm_keys=seed_niche_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm.benchmark()

            # Get the results table
            scib_type = bm.get_results(min_max_scale=False)
            biocons_scores = scib_type["Bio conservation"]
            batchcorr_scores = scib_type["Batch correction"]

            # Append scores for each embedding
            for embedding in seed_niche_keys:
                dict_DE[embedding].append(biocons_scores[embedding])
                dict_BC[embedding].append(batchcorr_scores[embedding])

        # Convert dictionaries to DataFrames and save as CSV files
        df_DE = pd.DataFrame(dict_DE).set_index("cell_type")
        df_BC = pd.DataFrame(dict_BC).set_index("cell_type")

        path_to_save_scib = setup.FIGURES_FOLDER
        df_DE.to_csv(os.path.join(path_to_save_scib, f"BioCons_cell_type_robustness_val_seed{bench_seed}.csv"))
        df_BC.to_csv(os.path.join(path_to_save_scib, f"BatchCorr_cell_type_robustness_val_seed{bench_seed}.csv"))
        print(f"[green]✓ Saved cell type niche analysis results for seed {bench_seed} (validation set)[/green]")

# ===== Ablation-only fixed sample benchmarking =====
# Runs when --ablation-only flag is set (compute_scib is False but we still need to benchmark ablations)
if args.ablation_only and latent_keys_ablation:
    import warnings
    warnings.filterwarnings("ignore")

    batchcorr = BatchCorrection(
        bras=False, ilisi_knn=True, kbet_per_label=False,
        graph_connectivity=False, pcr_comparison=False,
    )
    biocons = BioConservation(
        isolated_labels=True, nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False, silhouette_label=True, clisi_knn=True,
    )

    FIXED_SAMPLE_SEED = 34
    if scib_proportion < 1.0:
        adata_fixed_sample = sc.pp.sample(
            adata, fraction=scib_proportion, copy=True, replace=False, rng=FIXED_SAMPLE_SEED
        )
        print(f"Fixed sample: {adata_fixed_sample.n_obs} cells (sampled with seed {FIXED_SAMPLE_SEED})")
    else:
        adata_fixed_sample = adata.copy()

    print("\n[bold cyan]Benchmarking ABLATION NicheVI embeddings on fixed sample...[/bold cyan]")
    bm_fixed_ablation = Benchmarker(
        adata_fixed_sample,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,
        embedding_obsm_keys=latent_keys_ablation,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm_fixed_ablation.benchmark()
    bm_fixed_ablation.plot_results_table(min_max_scale=False, show=False)
    plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_ablation.png", bbox_inches="tight", dpi=800)
    plt.savefig(f"{setup.FIGURES_FOLDER}scib_fixed_sample_ablation.svg", bbox_inches="tight", dpi=800)
    scib_fixed_ablation_df = bm_fixed_ablation.get_results(min_max_scale=False, clean_names=True)
    scib_fixed_ablation_df.to_csv(f"{setup.FIGURES_FOLDER}scib_fixed_sample_results_ablation.csv")
    print("[green]✓ Saved fixed sample ablation results[/green]")
