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

from params_positive import setup, niche_setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

scvi.settings.seed = 34
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"


# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("/home/nathanl/scviva_paper/merfish_brain/checkpoints/", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER_SYNTHETIC_GENES, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER_SYNTHETIC_GENES, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))
# adata.layers["counts"] = adata.raw.X.copy()
print(adata)
# ----------scVI----------#


scvi_is_trained = True
nichevi_is_trained = False
save_scvi = True
compute_umap = True
compute_scib = False
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

# ----------NicheVI----------#


for setting in niche_setup.keys():
    print("[bold green]" + setting + "[/bold green]")
    setup_dict = niche_setup[setting]

    # preprocessing function to populate adata.obsm with the keys 'neighborhood_composition',
    # 'qz1_m', 'qz1_var', 'niche_indexes', 'niche_distances', 'qz1_m_niche_knn', 'qz1_var_niche_knn', 'qz1_m_niche_ct',
    # 'qz1_var_niche_ct'

    path_to_save_nichevae = path_to_save + "/nichevae_" + setting + "_" + str(setup.N_EPOCHS_NICHEVI) + ".pt"

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
    if setup_dict["add_synthetic_type"]:
        if "synthetic_cell_type" not in adata.obs[setup.CELL_TYPE].cat.categories:
            adata.obs[setup.CELL_TYPE] = adata.obs[setup.CELL_TYPE].cat.add_categories("synthetic_cell_type")
    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": setup.CELL_TYPE,
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",  # WARNING changed if synthetic genes
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(
        adata,
        k_nn=setup.K_NN,
        **setup_kwargs,
    )

    if setup_dict["add_synthetic_type"]:
        adata.obsm["qz1_m_niche_ct"] = adata.obsm["qz1_m_niche_ct_w_synth"].copy()
        adata.obsm["neighborhood_composition"] = adata.obsm["neighborhood_composition_w_synth"].copy()

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
            continue

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
            save_anndata=True,
        )

    if nichevi_is_trained:
        nichevae = nichevi.nicheSCVI.load(
            dir_path=path_to_save_nichevae,
            adata=adata,
        )

    history_setup[setting] = nichevae.history
    print(nichevae.history.keys())
    adata.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(batch_size=1024)

    adata.obs[setting + "_compo_error"] = nichevae.get_composition_error(return_mean=False).cpu()
    adata.obs[setting + "_niche_error"] = nichevae.get_niche_error(return_mean=False).cpu()

    if compute_umap:
        sc.pp.neighbors(adata, use_rep=setting + "_X_nicheVI")
        sc.tl.umap(adata, min_dist=0.3)

        sc.pl.umap(
            adata,
            color=[setup.CELL_TYPE],
            frameon=False,
            ncols=1,
            # palette=cc.glasbey_dark,
            palette=PALETTE,
            show=False,
        )

        plt.savefig(f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES}{setting}_umap.png", bbox_inches="tight", dpi=1000)
        plt.savefig(f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES}{setting}_umap.svg", bbox_inches="tight", dpi=1000)

# --- Save history + latent space for benchmarking ---#

adata.layers["counts"] = csr_matrix(adata.layers["counts"])

print(adata)
print("Saving adata...")

adata.write_h5ad(
    os.path.join(path_to_save_nichevae, data_file_name + "_nicheVI.h5ad"),
)

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
        bras=True,
        ilisi_knn=True,
        kbet_per_label=True,
        graph_connectivity=True,
        pcr_comparison=False,
    )

    biocons = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )

    from harmony import harmonize

    adata.obsm["X_banksy_02_harmony"] = harmonize(
        adata.obsm["banksy_pc_20_lambda_0.2"], adata.obs, batch_key=setup.BATCH
    )

    embedding_obsm_keys = (
        [latent_key]
        + [setting + "_X_nicheVI" for setting in niche_setup.keys()]
        + [
            # "simvi_both_08",
            "banksy_pc_20_lambda_0.2",
            # "simvi_interact_08",
            # "simvi_intrinsic_08",
            "X_banksy_02_harmony",
            # "X_resolvi",
            #  "X_banksy_04_harmony",
            "X_nicheformer",
            "X_nicheformer_e5",
            "simvi25_intrinsic_mae_50",
            # "simvi25_interact_mae_50",
            "simvi25_both_mae_50",
            "simvi25_all_mae_50",
        ]
    )

    if scib_proportion < 1.0:
        adata_sampled = sc.pp.sample(adata, fraction=scib_proportion, copy=True, replace=False, rng=34)
    else:
        adata_sampled = adata.copy()

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

    plt.savefig(f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES}scib_cell_type.png", bbox_inches="tight", dpi=800)
    plt.savefig(f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES}scib_cell_type.svg", bbox_inches="tight", dpi=800)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES}scib_cell_type_results.csv")

    TRESHOLD = setup.TRESHOLD

    adata.obs["cell_type_niche"] = adata.obs[setup.CELL_TYPE].astype(str) + "_" + adata.obs[setup.NICHE].astype(str)
    value_counts = adata.obs["cell_type_niche"].value_counts()
    cell_types_to_keep = value_counts[value_counts >= TRESHOLD].index
    print("n_obs: ", adata.n_obs)
    adata_filtered = adata[adata.obs["cell_type_niche"].isin(cell_types_to_keep)].copy()
    print("n_obs_filtered: ", adata_filtered.n_obs)

    # types_to_shuffle = [
    #     "astrocyte",
    #     "endothelial cell",
    #     "oligodendrocyte",
    #     "microglial cell",
    #     "pericyte",
    #     "oligodendrocyte_precursor cell",
    #     "GABAergic neuron",
    #     "glutamatergic neuron",
    # ]

    # for type in types_to_shuffle:
    #     adata_subset = adata_filtered[
    #         adata_filtered.obs[setup.CELL_TYPE] == type
    #     ].copy()

    #     for key in embedding_obsm_keys:
    #         clisi_key = _lisi_per_cell_type(
    #             adata_subset,
    #             key,
    #             setup.NICHE,
    #         )

    #         np.save(
    #             f"{setup.FIGURES_FOLDER_SYNTHETIC_GENES_SHUFFLE}cLISI_{type.split()[0]}_{key}.npy",
    #             clisi_key,
    #         )
