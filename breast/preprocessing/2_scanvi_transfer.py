import scanpy as sc
import scvi
import anndata as ad
import numpy as np

import matplotlib.pyplot as plt

adata_proseg = sc.read_h5ad("proseg_xenium_breast_cancer_S1_R1_2.h5ad")
adata_orig = sc.read_h5ad("../xenium_breast_cancer_S1_R1_2.h5ad")

models_history = {}

scanvi_ref_path = "scanvi_ref_S1_R1_2"


# REFERENCE:

trained_ref = True
trained_query = False

if trained_ref:
    scvivae = scvi.model.SCANVI.load("scanvi_ref_S1_R1_2", adata=adata_orig)
    # scanvi_query = scvi.model.SCANVI.load("scanvi_query_S1_R1_2")

    # models_history["scanvi_ref"] = scvivae.history
    # models_history["scanvi_query"] = scanvi_query.history


else:
    scvi.model.SCANVI.setup_anndata(
        adata_orig,
        layer="counts",
        unlabeled_category="ignore",
        batch_key="sample",
        labels_key="cell_type",
    )

    scvivae = scvi.model.SCANVI(
        adata_orig,
        gene_likelihood="poisson",
        n_layers=2,
        n_latent=10,
        linear_classifier=True,  # seems to improve scib?
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
    )

    scvivae.train(
        max_epochs=1000,
        train_size=0.8,
        validation_size=0.2,
        batch_size=512,
        plan_kwargs=dict(
            lr=1e-4,
            n_epochs_kl_warmup=400,
        ),
        # trainer_kwargs=dict(check_val_every_n_epoch=1),
        early_stopping=True,
    )

    scvivae.save(scanvi_ref_path, overwrite=True)

SCANVI_LATENT_KEY = "X_scANVI"

adata_orig.obsm[SCANVI_LATENT_KEY] = scvivae.get_latent_representation()

models_history["scanvi_ref"] = scvivae.history

# QUERY:
# Prepare data for query integration.
# This function will return a new AnnData object with padded zeros for missing features, as well as correctly sorted features.
scvi.model.SCANVI.prepare_query_anndata(adata_proseg, scvivae)

# Online update of a reference model with scArches algorithm:
if trained_query:
    scanvi_query = scvi.model.SCANVI.load("scanvi_query_S1_R1_2", adata=adata_proseg)


else:
    scanvi_query = scvi.model.SCANVI.load_query_data(adata_proseg, scanvi_ref_path)

    surgery_epochs = 1000
    train_kwargs_surgery = {
        "early_stopping": True,
        "early_stopping_monitor": "elbo_train",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "plan_kwargs": {"weight_decay": 0.0, "lr": 1e-4},  # "n_epochs_kl_warmup": 400},
    }

    scanvi_query.train(max_epochs=surgery_epochs, **train_kwargs_surgery)

    scanvi_query.save("scanvi_query_S1_R1_2", overwrite=True)

models_history["scanvi_query"] = scanvi_query.history

SCANVI_PREDICTIONS_KEY = "predictions_scanvi"

adata_proseg.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
adata_proseg.obs[SCANVI_PREDICTIONS_KEY] = scanvi_query.predict()


adata_full = ad.concat([adata_proseg, adata_orig], label="batch", join="outer")


adata_full.obs["batch"] = adata_full.obs["batch"].cat.rename_categories(
    ["Query-proseg", "Reference-xenium"]
)


# adata_full.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation(adata_full)

full_predictions = scanvi_query.predict(adata_full)
orig_predictions = scanvi_query.predict(adata_orig)
print(f"Accuracy: {np.mean(orig_predictions == adata_orig.obs['cell_type'])}")

adata_full.obs[SCANVI_PREDICTIONS_KEY] = full_predictions


adata_full.obsm[SCANVI_LATENT_KEY + "_MDE"] = scvi.model.utils.mde(
    adata_full.obsm[SCANVI_LATENT_KEY]
)

adata_full.write_h5ad("query_ref_S1_R1_2.h5ad")


sc.pl.embedding(
    adata_full,
    basis=SCANVI_LATENT_KEY + "_MDE",
    # color=[setup.CELL_TYPE],
    color=[SCANVI_PREDICTIONS_KEY, "batch"],
    frameon=False,
    ncols=1,
    palette="tab20",
    show=False,
)

plt.savefig("figures/full_mde.png", bbox_inches="tight", dpi=1000)


for key in models_history.keys():
    list_of_registered_keys = models_history[key].keys()
    keys_to_plot = (
        [keys for keys in list_of_registered_keys if "validation" in keys]
        + ["kl_weight"]
        if "kl_weight" in list_of_registered_keys
        else [keys for keys in list_of_registered_keys if "validation" in keys]
    )
    series_to_plot = [models_history[key][keys] for keys in keys_to_plot]

    n_plots = len(series_to_plot)

    fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 5 * n_plots))

    # Flatten the axs array to access each subplot individually
    axs = axs.flatten()

    # Plot each series in a subplot
    for i, ax in enumerate(axs[:n_plots]):
        if i < len(series_to_plot):
            series_to_plot[i].plot(ax=ax)
            ax.set_title(keys_to_plot[i])

    # MDE_KEY = key + "_MDE"

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save the figure to a file
    plt.savefig(f"figures/{key}_history.png", dpi=80, bbox_inches="tight")
