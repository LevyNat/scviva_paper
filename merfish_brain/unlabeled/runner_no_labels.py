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

from params import setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

scvi.settings.seed = 34
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"

# you can't plots UMAPS otherwise...
sc.settings._vector_friendly = True


# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

# path_to_save = os.path.join("../checkpoints", data_file_name)
path_to_save = f"/home/nathanl/scviva_paper/merfish_brain/checkpoints/{data_file_name}"
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER_SHUFFLE, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))


adata.layers["counts"] = adata.raw.X.copy()
print(adata)

print(
    "Will save the adata as: ",
    os.path.join(path_to_save, data_file_name + "_nolabel_nicheVI.h5ad"),
)


# ----------scVI----------#


scvi_is_trained = True
nichevi_is_trained = True
save_scvi = True
compute_umap = True
compute_scib = True
scib_proportion = 0.3

PALETTE = "tab20"

setup.EXPRESSION_MODEL = "scvi"

print(
    f"{setup.EXPRESSION_MODEL} trained: ",
    scvi_is_trained,
    "nicheVI trained: ",
    nichevi_is_trained,
    "compute UMAP: ",
    compute_umap,
)

history_setup = {}

if scvi_is_trained is False:
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
        scvivae.save(
            dir_path=path_to_save
            + f"/{setup.EXPRESSION_MODEL}vae_E"
            + str(setup.N_EPOCHS_SCVI)
            + "_"
            + str(setup.LIKELIHOOD)
            # + "L"
            + ".pt",
            save_anndata=False,
            overwrite=True,
        )

if scvi_is_trained:
    scvivae = scvi.model.SCVI.load(
        dir_path=path_to_save
        + f"/{setup.EXPRESSION_MODEL}vae_E"
        + str(setup.N_EPOCHS_SCVI)
        + "_"
        + str(setup.LIKELIHOOD)
        + ".pt",
        adata=adata,
    )

latent_key = "X_scVI"
adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)


history_setup[latent_key] = scvivae.history

print(scvivae.history.keys())


print(f"{setup.EXPRESSION_MODEL} done...")

# Build neighbor graph on X_scanvi
sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15, key_added="scvi_neighbors")

# Sweep resolutions and print cluster counts
LEIDEN_RES = list(setup.LEIDEN_RES)
for res in LEIDEN_RES:
    sc.tl.leiden(
        adata,
        resolution=res,
        neighbors_key="scvi_neighbors",
        key_added=f"leiden_scvi_res{res}",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    n = adata.obs[f"leiden_scvi_res{res}"].nunique()
    print(f"res={res}: {n} clusters")

if compute_umap:
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata, min_dist=0.3)

    sc.pl.umap(
        adata,
        color=[setup.CELL_TYPE] + [f"leiden_scvi_res{r}" for r in setup.FIGURE_LEIDEN_RES],
        frameon=False,
        ncols=1,
        # palette=cc.glasbey_dark,
        palette=PALETTE,
        show=False,
    )

    plt.savefig(f"{setup.FIGURES_FOLDER}{latent_key}_umap_nolabel.png", bbox_inches="tight", dpi=1000)
    plt.savefig(f"{setup.FIGURES_FOLDER}{latent_key}_umap_nolabel.svg", bbox_inches="tight", dpi=1000)


# ----------NicheVI----------#
# ----- Base NicheVI configuration -----#
def make_niche_config(
    leiden_res: float,
    spatial_weight: float = 10,
    niche_rec_weight: float = 10,
    compo_rec_weight: float = 10,
) -> dict:
    """Create a NicheVI configuration"""
    return {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": spatial_weight,
        "niche_rec_weight": niche_rec_weight,
        "compo_rec_weight": compo_rec_weight,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": setup.N_LAYERS_NICHE,
        "n_layers_compo": setup.N_LAYERS_COMPO,
        "n_hidden_niche": setup.N_HIDDEN_NICHE,
        "n_hidden_compo": setup.N_HIDDEN_COMPO,
        "niche_expression": "scvi",
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "prior_mixture": False,
        "prior_mixture_k": 1,
        "label_key": f"leiden_scvi_res{leiden_res}",
    }


for res in LEIDEN_RES:
    print("[bold green]" + str(res) + "[/bold green]")
    setup_dict = make_niche_config(leiden_res=res)

    # preprocessing function to populate adata.obsm with the keys 'neighborhood_composition',
    # 'qz1_m', 'qz1_var', 'niche_indexes', 'niche_distances', 'qz1_m_niche_knn', 'qz1_var_niche_knn', 'qz1_m_niche_ct',
    # 'qz1_var_niche_ct'

    setting = f"{res}_s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}"

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

    n_neighbors = setup_dict.get("n_neighbors", setup.K_NN)

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": setup_dict["label_key"],
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(
        adata,
        k_nn=n_neighbors,
        **setup_kwargs,
    )

    nichevi.nicheSCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=setup.BATCH,
        **setup_kwargs,
    )

    # if nichevi_is_trained is False:
    # check if path_to_save_nichevae exists
    if os.path.exists(path_to_save_nichevae):
        print("Model already exists. Loading...")
        nichevae = nichevi.nicheSCVI.load(
            dir_path=path_to_save_nichevae,
            adata=adata,
        )
    else:
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
        )

    # if nichevi_is_trained:
    #     nichevae = nichevi.nicheSCVI.load(
    #         dir_path=path_to_save_nichevae,
    #         adata=adata,
    #     )

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

        plt.savefig(f"{setup.FIGURES_FOLDER}{setting}_umap.png", bbox_inches="tight", dpi=1000)
        plt.savefig(f"{setup.FIGURES_FOLDER}{setting}_umap.svg", bbox_inches="tight", dpi=1000)

# --- Save history + latent space for benchmarking ---#

adata.layers["counts"] = csr_matrix(adata.layers["counts"])

print(adata)
print("Saving adata...")

adata.write_h5ad(
    os.path.join(path_to_save, data_file_name + "_nolabel_nicheVI.h5ad"),
)

# adata.write_h5ad(os.path.join(data_dir, data_file))

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

    # from harmony import harmonize

    # adata.obsm["X_banksy_02_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.2"], adata.obs, batch_key=setup.BATCH
    # )

    embedding_obsm_keys = (
        # [latent_key]
        # [f"{0.15}_s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}" + "_X_nicheVI"]
        []
        + [
            f"{res}_s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}" + "_X_nicheVI"
            for res in LEIDEN_RES
        ]
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

    # plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_ablation_k.png", bbox_inches="tight", dpi=300)
    # plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_ablation_k.svg", bbox_inches="tight", dpi=300)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{setup.FIGURES_FOLDER}scib_cell_type_results_ablation_nolabel.csv")

    # effect on region labels: region conservation per Leiden cluster, for each resolution shown in Fig. S2
    for SELECTED_RES in setup.FIGURE_LEIDEN_RES:
        pseudo_label_key = f"leiden_scvi_res{SELECTED_RES}"
        TRESHOLD = 400

        # Keep only pseudo-cluster × region combinations with enough cells
        adata.obs["pseudo_type_niche"] = adata.obs[pseudo_label_key].astype(str) + "_" + adata.obs[setup.NICHE].astype(str)

        value_counts = adata.obs["pseudo_type_niche"].value_counts()
        pseudo_types_to_keep = value_counts[value_counts >= TRESHOLD].index

        adata_subset = adata[adata.obs["pseudo_type_niche"].isin(pseudo_types_to_keep)].copy()

        # Only iterate over pseudo-clusters that survive filtering
        pseudo_clusters = adata_subset.obs[pseudo_label_key].unique().tolist()

        embedding_obsm_keys_region = [
            f"{SELECTED_RES}_s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}" + "_X_nicheVI"
        ]

        dict_DE = {"pseudo_cluster": []}
        dict_BC = {"pseudo_cluster": []}

        for embedding in embedding_obsm_keys_region:
            dict_DE[embedding] = []
            dict_BC[embedding] = []

        for cluster in pseudo_clusters:
            adata_subset_cluster = adata_subset[adata_subset.obs[pseudo_label_key] == cluster].copy()

            # Check that this pseudo-cluster spans at least 2 true regions
            if adata_subset_cluster.obs[setup.NICHE].nunique() < 2:
                print(f"Skipping {cluster} as it only has one region")
                continue

            print(cluster)

            dict_DE["pseudo_cluster"].append(cluster)
            dict_BC["pseudo_cluster"].append(cluster)

            bm = Benchmarker(
                adata_subset_cluster,
                batch_key=setup.BATCH,
                label_key=setup.NICHE,  # true region labels for evaluation
                embedding_obsm_keys=embedding_obsm_keys_region,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=batchcorr,
                n_jobs=-1,
            )
            bm.benchmark()

            scib_type = bm.get_results(min_max_scale=False)
            biocons_scores = scib_type["Bio conservation"]
            batchcorr_scores = scib_type["Batch correction"]

            for embedding in embedding_obsm_keys_region:
                dict_DE[embedding].append(biocons_scores[embedding])
                dict_BC[embedding].append(batchcorr_scores[embedding])

        df_DE = pd.DataFrame(dict_DE).set_index("pseudo_cluster")
        df_BC = pd.DataFrame(dict_BC).set_index("pseudo_cluster")

        path_to_save_scib = setup.FIGURES_FOLDER
        df_DE.to_csv(os.path.join(path_to_save_scib, f"BioCons_cell_type_ablation_nolabel_res{SELECTED_RES}.csv"))
        df_BC.to_csv(os.path.join(path_to_save_scib, f"BatchCorr_cell_type_ablation_nolabel_res{SELECTED_RES}.csv"))
