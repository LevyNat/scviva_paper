import scanpy as sc
import scvi
import anndata as ad
import numpy as np

import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional

from scvi.external import RESOLVI

from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    # DATA_FOLDER: str = (
    #     "/home/labs/nyosef/Collaboration/SpatialDatasets/xenium/human_breast_cancer/"
    # )
    DATA_FOLDER: str = "/home/nathanlevy/sda/Data/"
    DATA_FILE: str = "xenium_breast_cancer_S1_R1_2.h5ad"
    DATA_FILE_PROSEG: str = "proseg_scanvi_xenium_breast_cancer_S1_R1_2.h5ad"

    FIGURES_FOLDER: str = "figures/"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "sample"
    SAMPLE: str = "sample"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "kmeans_alpha_5"

    N_EPOCHS = 100
    BATCH_SIZE = 1024


setup = PARAMS()
trained_ref = True

adata_proseg = sc.read_h5ad(setup.DATA_FOLDER + setup.DATA_FILE_PROSEG)
# adata_proseg.obsm["spatial"] = adata_proseg.obs[["x", "y"]].values
# adata_proseg.write_h5ad("proseg_xenium_breast_cancer_S1_R1_2.h5ad")

print(adata_proseg)

print(adata_proseg.layers["counts"].sum(axis=1).min())

adata_proseg.X = csr_matrix(adata_proseg.X)
adata_proseg.layers["counts"] = csr_matrix(adata_proseg.layers["counts"])

adata_orig = sc.read_h5ad(setup.DATA_FOLDER + setup.DATA_FILE)


adata_orig.X = csr_matrix(adata_orig.X)
adata_orig.layers["counts"] = csr_matrix(adata_orig.layers["counts"])

# adata_orig.obs.reset_index(inplace=True)
# adata_proseg.obs.reset_index(inplace=True)

models_history = {}

scanvi_ref_path = "scanvi_ref_S1_R1_2"
model_path = "resolvae"


if not trained_ref:
    RESOLVI.setup_anndata(
        adata_orig,
        layer="counts",
        batch_key=setup.BATCH,
        labels_key=setup.CELL_TYPE,
        unlabeled_category="ignore",
        prepare_data_kwargs={"n_neighbors": 20, "spatial_rep": setup.COORDS},
    )

    resolvae = RESOLVI(
        adata_orig,
        n_latent=10,
        n_hidden=32,
        n_hidden_encoder=128,
        semisupervised=True,
        mixture_k=50,
        deeply_inject_covariates=True,
        encode_covariates=False,
        conditional_norm="none",
    )
    extra_lr_parameters = ["per_neighbor_diffusion_map", "u_prior_means"]

    resolvae.train(
        max_epochs=setup.N_EPOCHS,
        # train_size=0.9,
        # validation_size=0.1,
        lr=3e-3,
        lr_extra=1e-2,
        extra_lr_parameters=extra_lr_parameters,
        batch_size=setup.BATCH_SIZE,
        enable_progress_bar=True,
        n_epochs_kl_warmup=400,
        early_stopping=True,
        early_stopping_monitor="elbo_train",
    )

    resolvae.save(model_path + "_" + str(setup.N_EPOCHS), overwrite=True)

else:
    resolvae = RESOLVI.load(model_path + "_" + str(setup.N_EPOCHS), adata_orig)

RESOLVI_LATENT_KEY = "X_resolvi"

adata_orig.obsm[RESOLVI_LATENT_KEY] = resolvae.get_latent_representation()

models_history["resolvi_ref"] = resolvae.history

RESOLVI.prepare_query_anndata(adata_proseg, resolvae)
# adata_proseg = adata_proseg[adata_proseg.layers["counts"].sum(1) > 5].copy()
adata_proseg.obs_names = ["query_" + i for i in adata_proseg.obs_names]
adata_proseg.obs_names_make_unique()
resolvi_transfer = RESOLVI.load_query_data(
    adata_proseg,
    resolvae,
)
adata_proseg.obsm["celltype_transfer_resolvi"] = resolvi_transfer.predict(
    adata=adata_proseg, soft=True, num_samples=3, batch_size=10000
)
adata_proseg.obs["predicted_celltype_resolvi"] = adata_proseg.obsm[
    "celltype_transfer_resolvi"
].idxmax(axis=1)
adata_proseg.obs["predicted_celltype_prob_resolvi"] = adata_proseg.obsm[
    "celltype_transfer_resolvi"
].max(axis=1)

# repeat for adata_orig:

# adata_orig.obsm["celltype_transfer_resolvi"] = resolvi_transfer.predict(
#     adata=adata_orig, soft=True, num_samples=3, batch_size=10000
# )
# adata_orig.obs["predicted_celltype_resolvi"] = adata_orig.obsm[
#     "celltype_transfer_resolvi"
# ].idxmax(axis=1)
# adata_orig.obs["predicted_celltype_prob_resolvi"] = adata_orig.obsm[
#     "celltype_transfer_resolvi"
# ].max(axis=1)

adata_proseg.obsm[RESOLVI_LATENT_KEY] = resolvi_transfer.get_latent_representation()


# adata_full = ad.concat([adata_proseg, adata_orig], label="batch", join="outer")

# adata_full.obs["batch"] = adata_full.obs["batch"].cat.rename_categories(
#     ["Query-proseg", "Reference-xenium"]
# )

# orig_predictions = resolvi_transfer.predict(adata_orig, soft=False)
# print(
#     f"Accuracy on Origin: {np.mean(orig_predictions == adata_orig.obs[setup.CELL_TYPE])}"
# )

print(
    f"Accuracy w/r scanVI on Proseg: {np.mean(adata_proseg.obs['predicted_celltype_resolvi'] == adata_proseg.obs['predictions_scanvi'])}"
)


adata_proseg.obsm[RESOLVI_LATENT_KEY + "_MDE"] = scvi.model.utils.mde(
    adata_proseg.obsm[RESOLVI_LATENT_KEY]
)

# adata_full.write_h5ad(setup.DATA_FOLDER + "resolvi_query_ref_S1_R1_2.h5ad")
adata_proseg.write_h5ad(
    setup.DATA_FOLDER + "proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad"
)


CMAP = "Paired"

sc.pl.embedding(
    adata_proseg,
    basis=RESOLVI_LATENT_KEY + "_MDE",
    # color=[setup.CELL_TYPE],
    color=[
        "predicted_celltype_resolvi",
        "predictions_scanvi",
        "predicted_celltype_prob_resolvi",
        # "batch",
    ],
    frameon=False,
    ncols=1,
    palette=CMAP,
    show=False,
)

plt.savefig("figures_resolvi/proseg_mde.png", bbox_inches="tight", dpi=100)

for sample in adata_proseg.obs["sample"].unique()[:]:
    # with plt.rc_context():
    sc.pl.spatial(
        adata_proseg[adata_proseg.obs[setup.SAMPLE] == sample],
        spot_size=40,
        color=[
            "predicted_celltype_resolvi",
            "predictions_scanvi",
            # "predicted_celltype_prob_resolvi",
            # "batch",
        ],
        ncols=1,
        frameon=False,
        title=sample,
        cmap=CMAP,
        show=False,
    )

    plt.savefig("figures_resolvi/spatial_breast.png", bbox_inches="tight", dpi=100)

for key in models_history.keys():
    list_of_registered_keys = models_history[key].keys()
    keys_to_plot = [keys for keys in list_of_registered_keys]
    print(keys_to_plot)
    series_to_plot = [models_history[key][keys] for keys in keys_to_plot]

    # n_plots = len(series_to_plot)

    # fig, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 5 * n_plots))

    # # Flatten the axs array to access each subplot individually
    # axs = axs.flatten()

    # # Plot each series in a subplot
    # for i, ax in enumerate(axs[:n_plots]):
    #     if i < len(series_to_plot):
    #         series_to_plot[i].plot(ax=ax)
    #         ax.set_title(keys_to_plot[i])

    # # MDE_KEY = key + "_MDE"

    series_to_plot[0].plot()

    plt.title(keys_to_plot[0])

    # Adjust the spacing between subplots
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save the figure to a file
    plt.savefig(f"figures_resolvi/{key}_history.png", dpi=80, bbox_inches="tight")
