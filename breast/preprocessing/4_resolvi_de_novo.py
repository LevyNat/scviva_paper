import scanpy as sc
import scvi
import anndata as ad
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from dataclasses import dataclass
from typing import Optional

from scvi.external import RESOLVI


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    # DATA_FOLDER: str = (
    #     "/home/labs/nyosef/Collaboration/SpatialDatasets/xenium/human_breast_cancer/"
    # )
    DATA_FOLDER: str = "/home/nathanlevy/sda/Data/"
    DATA_FILE: str = "xenium_breast_cancer_S1_R1_2.h5ad"
    DATA_FILE_PROSEG: str = "proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad"

    FIGURES_FOLDER: str = "figures/"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "sample"
    SAMPLE: str = "sample"
    CELL_TYPE: str = "predicted_celltype_resolvi"
    NICHE: str = "kmeans_alpha_5"

    N_EPOCHS = 100
    BATCH_SIZE = 1024


setup = PARAMS()

adata_proseg = sc.read_h5ad(setup.DATA_FOLDER + setup.DATA_FILE_PROSEG)
# adata_proseg.obsm["spatial"] = adata_proseg.obs[["x", "y"]].values
# adata_proseg.write_h5ad("proseg_xenium_breast_cancer_S1_R1_2.h5ad")

print(adata_proseg)
# adata_orig = sc.read_h5ad(setup.DATA_FOLDER + setup.DATA_FILE_PROSEG)

disagrement = (
    adata_proseg.obs["predictions_scanvi"]
    != adata_proseg.obs["predicted_celltype_resolvi"]
)


adata_disagree = adata_proseg.copy()

# Ensure the column is categorical
cell_type_column = adata_disagree.obs[setup.CELL_TYPE]
if not pd.api.types.is_categorical_dtype(cell_type_column):
    cell_type_column = cell_type_column.astype("category")

# Add 'Disagreement' category if it does not exist
if "Disagreement" not in cell_type_column.cat.categories:
    cell_type_column = cell_type_column.cat.add_categories(["Disagreement"])

# Replace the categories based on the boolean index
cell_type_column[disagrement] = "Disagreement"

# Assign the modified column back to the AnnData object
adata_disagree.obs[setup.CELL_TYPE] = cell_type_column


RESOLVI.setup_anndata(
    adata_disagree,
    layer="counts",
    batch_key=setup.BATCH,
    labels_key=setup.CELL_TYPE,
    unlabeled_category="Disagreement",
    prepare_data_kwargs={"n_neighbors": 20, "spatial_rep": setup.COORDS},
)

resolvae = RESOLVI(
    adata_disagree,
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

resolvae.save("resolvae_proseg_" + str(setup.N_EPOCHS), overwrite=True)

RESOLVI_LATENT_KEY = "X_resolvi_proseg"

adata_proseg.obsm[RESOLVI_LATENT_KEY] = resolvae.get_latent_representation()

adata_disagree.obs["predictions_resolvi_proseg"] = resolvae.predict()

adata_proseg.obs["predictions_resolvi_proseg"] = adata_disagree.obs[
    "predictions_resolvi_proseg"
].values

adata_proseg.write_h5ad(setup.DATA_FOLDER + setup.DATA_FILE_PROSEG)
