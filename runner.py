import argparse
import importlib.util
import os
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import nichevi
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from rich import print
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scipy.sparse import csr_matrix

REPO_ROOT = Path(__file__).resolve().parent

scvi.settings.seed = 34
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"

PALETTE = "tab20"


def load_params(dataset_arg):
    dataset_dir = Path(dataset_arg)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()

    spec = importlib.util.spec_from_file_location("params", dataset_dir / "params.py")
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)

    setup = params_module.setup
    niche_setup = getattr(params_module, "niche_setup", {})
    return setup, niche_setup


def main(setup, niche_setup):
    force_retrain = getattr(setup, "FORCE_RETRAIN", False)
    compute_umap = getattr(setup, "COMPUTE_UMAP", True)
    compute_scib = getattr(setup, "COMPUTE_SCIB", True)
    scib_proportion = getattr(setup, "SCIB_PROPORTION", 0.3)

    # ----------load data----------#

    data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE
    data_file_name = os.path.splitext(data_file)[0]
    print(data_file_name)

    path_to_save = os.path.join(setup.CHECKPOINT_FOLDER, data_file_name)
    # OUTPUT_TAG lets several experiments share one checkpoint folder (same trained models)
    # without overwriting each other's output h5ad / scib tables. Default "" = unchanged names.
    run_name = data_file_name + getattr(setup, "OUTPUT_TAG", "")
    os.makedirs(path_to_save, exist_ok=True)
    os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)

    adata = ad.read_h5ad(os.path.join(data_dir, data_file))

    if getattr(setup, "COUNTS_LAYER_FROM_RAW", True):
        adata.layers["counts"] = adata.raw.X.copy()

    min_counts = getattr(setup, "MIN_COUNTS_PER_CELL", None)
    if min_counts is not None:
        counts = adata.layers["counts"]
        cell_counts = counts.toarray().sum(axis=1) if hasattr(counts, "toarray") else counts.sum(axis=1)
        n_cells_before = adata.n_obs
        adata = adata[cell_counts >= min_counts].copy()
        print(f"Filtered {n_cells_before - adata.n_obs} cells with < {min_counts} counts")

    print(adata)
    print(
        "Will save the adata as: ",
        os.path.join(path_to_save, run_name + "_nicheVI.h5ad"),
    )

    # ----------scVI----------#

    print(
        f"{setup.EXPRESSION_MODEL} force retrain: ",
        force_retrain,
        "compute UMAP: ",
        compute_umap,
    )

    history_setup = {}

    scvi_checkpoint_path = (
        path_to_save
        + f"/{setup.EXPRESSION_MODEL}vae_E"
        + str(setup.N_EPOCHS_SCVI)
        + "_"
        + str(setup.LIKELIHOOD)
        + ".pt"
    )

    if os.path.exists(scvi_checkpoint_path) and not force_retrain:
        print("scVI/SCANVI model already exists. Loading...")
        if setup.EXPRESSION_MODEL == "scanvi":
            scvivae = scvi.model.SCANVI.load(dir_path=scvi_checkpoint_path, adata=adata)
        if setup.EXPRESSION_MODEL == "scvi":
            scvivae = scvi.model.SCVI.load(dir_path=scvi_checkpoint_path, adata=adata)
    else:
        if setup.EXPRESSION_MODEL == "scanvi":
            scvi.model.SCANVI.setup_anndata(
                adata,
                layer="counts",
                unlabeled_category="ignore",
                batch_key=setup.BATCH,
                labels_key=setup.CELL_TYPE,
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
            early_stopping=True,
        )

        scvivae.save(dir_path=scvi_checkpoint_path, save_anndata=False)

    latent_key = "X_scVI" if setup.EXPRESSION_MODEL == "scvi" else "X_scanvi"

    adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)

    if compute_umap:
        _plot_umap(adata, latent_key, setup, f"{setup.FIGURES_FOLDER}{latent_key}_umap")

    history_setup[latent_key] = scvivae.history

    print(scvivae.history.keys())
    print(f"{setup.EXPRESSION_MODEL} done...")

    # ----------NicheVI----------#

    for setting in niche_setup.keys():
        print("[bold green]" + setting + "[/bold green]")
        setup_dict = niche_setup[setting]

        path_to_save_nichevae = path_to_save + "/nichevae_" + setting + "_" + str(setup.N_EPOCHS_NICHEVI) + ".pt"

        niche_likelihood = "gaussian"
        if setup_dict["niche_expression"] == "pca":
            adata.obsm["qz1_m"] = adata.obsm["X_pca"]
        elif setup_dict["niche_expression"] == "scvi":
            adata.obsm["qz1_m"] = adata.obsm[latent_key]
        elif setup_dict["niche_expression"] == "resolvi":
            adata.obsm["qz1_m"] = adata.obsm["X_resolvi"].copy()
        elif setup_dict["niche_expression"] == "gaussian_counts":
            adata.obsm["qz1_m"] = adata.X.toarray()

        n_neighbors = setup_dict.get("n_neighbors", setup.K_NN)

        setup_kwargs = {
            "sample_key": setup.SAMPLE,
            "labels_key": setup.CELL_TYPE,
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

        if os.path.exists(path_to_save_nichevae) and not force_retrain:
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
                niche_likelihood=niche_likelihood,
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
                    n_epochs_spatial_warmup=getattr(setup, "SPATIAL_WARMUP", None),
                    min_spatial_weight=getattr(setup, "MIN_SPATIAL_WEIGHT", 1.0),
                    max_spatial_weight=getattr(setup, "MAX_SPATIAL_WEIGHT", 1.0),
                    optimizer=setup.OPTIMIZER,
                    weight_decay=setup.WEIGHT_DECAY,
                    reduce_lr_on_plateau=setup.REDUCE_LR_ON_PLATEAU,
                ),
                early_stopping_patience=100,  # trick because sometimes KL goes up.
            )

            nichevae.save(
                dir_path=path_to_save_nichevae,
                save_anndata=False,
            )

        history_setup[setting] = nichevae.history
        print(nichevae.history.keys())
        adata.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(
            batch_size=setup.BATCH_SIZE_NICHEVI
        )

        adata.obs[setting + "_compo_error"] = nichevae.get_composition_error(return_mean=False).cpu()
        adata.obs[setting + "_niche_error"] = nichevae.get_niche_error(return_mean=False).cpu()

        if compute_umap:
            _plot_umap(adata, setting + "_X_nicheVI", setup, f"{setup.FIGURES_FOLDER}{setting}_umap")

    # --- Save history + latent space for benchmarking ---#

    adata.layers["counts"] = csr_matrix(adata.layers["counts"])

    print(adata)
    print("Saving adata...")

    adata.write_h5ad(
        os.path.join(path_to_save, run_name + "_nicheVI.h5ad"),
    )

    # ----------Benchmarks----------#

    if compute_scib:
        _run_scib_benchmarks(adata, setup, niche_setup, latent_key, run_name, scib_proportion)


def _plot_umap(adata, use_rep, setup, save_path_no_ext):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata, min_dist=0.3)

    color_by = [setup.CELL_TYPE]
    if adata.obs[setup.BATCH].nunique() > 1:
        color_by.append(setup.BATCH)

    sc.pl.umap(
        adata,
        color=color_by,
        frameon=False,
        ncols=2,
        palette=PALETTE,
        show=False,
    )

    plt.savefig(f"{save_path_no_ext}.png", bbox_inches="tight", dpi=1000)
    plt.savefig(f"{save_path_no_ext}.svg", bbox_inches="tight", dpi=1000)


def _batch_correction_metrics_for(adata_subset, setup, batchcorr):
    n_batches = adata_subset.obs[setup.BATCH].nunique()
    if n_batches > 1:
        return batchcorr
    print(f"Only {n_batches} unique value(s) in batch key '{setup.BATCH}' — skipping batch-correction metrics.")
    return None


def _run_scib_benchmarks(adata, setup, niche_setup, latent_key, data_file_name, scib_proportion):
    warnings.filterwarnings("ignore")

    batchcorr = BatchCorrection(
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

    embedding_obsm_keys = []
    if getattr(setup, "INCLUDE_EXPRESSION_LATENT_IN_SCIB", True):
        embedding_obsm_keys.append(latent_key)
    embedding_obsm_keys += [setting + "_X_nicheVI" for setting in niche_setup.keys()]
    embedding_obsm_keys += list(getattr(setup, "EMBEDDING_OBSM_KEYS", []))

    if scib_proportion < 1.0:
        adata_sampled = sc.pp.sample(adata, fraction=scib_proportion, copy=True, replace=False, rng=34)
    else:
        adata_sampled = adata.copy()

    batch_correction_metrics = _batch_correction_metrics_for(adata_sampled, setup, batchcorr)

    bm = Benchmarker(
        adata_sampled,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batch_correction_metrics,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_{data_file_name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{setup.FIGURES_FOLDER}scib_cell_type_{data_file_name}.svg", bbox_inches="tight", dpi=300)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{setup.FIGURES_FOLDER}scib_cell_type_results_{data_file_name}.csv")

    # effect on region labels
    treshold = getattr(setup, "TRESHOLD", 400)
    adata.obs["cell_type_niche"] = adata.obs[setup.CELL_TYPE].astype(str) + "_" + adata.obs[setup.NICHE].astype(str)
    value_counts = adata.obs["cell_type_niche"].value_counts()
    cell_types_to_keep = value_counts[value_counts >= treshold].index
    adata_subset = adata[adata.obs["cell_type_niche"].isin(cell_types_to_keep)].copy()

    dict_DE = {"cell_type": []}
    dict_BC = {"cell_type": []}

    for embedding in embedding_obsm_keys:
        dict_BC[embedding] = []
        dict_DE[embedding] = []

    any_batch_correction = False

    for cell_type in adata_subset.obs[setup.CELL_TYPE].unique().tolist():
        adata_subset_type = adata_subset[adata_subset.obs[setup.CELL_TYPE] == cell_type].copy()
        type_niche_list = adata_subset_type.obs["cell_type_niche"].unique().tolist()

        if len(type_niche_list) == 1:
            print(f"Skipping {cell_type} as it only has one niche")
            continue

        print(cell_type)

        dict_DE["cell_type"].append(cell_type)

        type_batch_correction_metrics = _batch_correction_metrics_for(adata_subset_type, setup, batchcorr)

        bm = Benchmarker(
            adata_subset_type,
            batch_key=setup.BATCH,
            label_key="cell_type_niche",
            embedding_obsm_keys=embedding_obsm_keys,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=type_batch_correction_metrics,
            n_jobs=-1,
        )
        bm.benchmark()

        scib_type = bm.get_results(min_max_scale=False)
        biocons_scores = scib_type["Bio conservation"]

        for embedding in embedding_obsm_keys:
            dict_DE[embedding].append(biocons_scores[embedding])

        if type_batch_correction_metrics is not None:
            any_batch_correction = True
            dict_BC["cell_type"].append(cell_type)
            batchcorr_scores = scib_type["Batch correction"]
            for embedding in embedding_obsm_keys:
                dict_BC[embedding].append(batchcorr_scores[embedding])

    df_DE = pd.DataFrame(dict_DE).set_index("cell_type")

    path_to_save_scib = setup.FIGURES_FOLDER
    df_DE.to_csv(os.path.join(path_to_save_scib, f"BioCons_cell_type_{data_file_name}.csv"))

    df_BC = None
    if any_batch_correction:
        df_BC = pd.DataFrame(dict_BC).set_index("cell_type")
        df_BC.to_csv(os.path.join(path_to_save_scib, f"BatchCorr_cell_type_{data_file_name}.csv"))

    # ----------Plots----------#

    colors = _scib_plot_colors(embedding_obsm_keys, latent_key)
    _plot_grouped_bar_scib(
        df_DE, colors, "Bio cons", f"{setup.FIGURES_FOLDER}cell_type_scib_bio_cons_{data_file_name}"
    )
    if df_BC is not None:
        _plot_grouped_bar_scib(
            df_BC, colors, "Batch corr", f"{setup.FIGURES_FOLDER}cell_type_scib_batch_{data_file_name}"
        )
    _plot_bio_vs_batch_scatter(scib_df, colors, f"{setup.FIGURES_FOLDER}cell_type_scib_scatter_{data_file_name}")
    _plot_region_vs_celltype_scatter(
        df_DE, scib_df, colors, f"{setup.FIGURES_FOLDER}region_celltype_scatter_{data_file_name}"
    )


def _scib_plot_colors(embedding_obsm_keys, latent_key):
    """Assign a consistent color per method family, following the
    custom_colors convention (NICHEVI=red, expression model=green, BANKSY=blue/navy,
    SIMVI=orange shades, NICHEFORMER=teal/deepskyblue). Matched by substring rather than
    exact key so it holds across datasets whose exact key strings differ (e.g. a "scvi"
    vs "scanvi" run), assuming EMBEDDING_OBSM_KEYS stays a superset of these families.
    """
    colors = {}
    fallback_iter = iter(plt.get_cmap("tab10").colors)
    for key in embedding_obsm_keys:
        key_lower = key.lower()
        if key == latent_key:
            colors[key] = "green"
        elif "nichevi" in key_lower:
            colors[key] = "red"
        elif "banksy" in key_lower:
            colors[key] = "blue" if "harmony" in key_lower else "navy"
        elif "simvi" in key_lower:
            if "intrinsic" in key_lower:
                colors[key] = "darkorange"
            elif "interact" in key_lower:
                colors[key] = "orangered"
            else:
                colors[key] = "orange"  # e.g. "both"/"all"
        elif "nicheformer" in key_lower:
            colors[key] = "teal" if key_lower.endswith("_e5") else "deepskyblue"
        else:
            colors[key] = next(fallback_iter, "gray")
    return colors


def _plot_grouped_bar_scib(df, colors, title, save_path_no_ext, group_spacing=1.3, bar_width=0.15):
    n_methods = len(df.columns)
    fig, ax = plt.subplots(figsize=(len(df) * group_spacing * 0.9, 5))

    x = np.arange(len(df.index)) * group_spacing
    for i, method in enumerate(df.columns):
        ax.bar(x + i * bar_width, df[method], width=bar_width, label=method, color=colors.get(method))

    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(df.index, rotation=90)
    ax.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylabel("Scores")
    ax.set_xlabel("Cell Types")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{save_path_no_ext}.png", dpi=1000, bbox_inches="tight")
    plt.savefig(f"{save_path_no_ext}.svg", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def _plot_bio_vs_batch_scatter(scib_df, colors, save_path_no_ext):
    df = scib_df.drop(index="Metric Type", errors="ignore")

    if "Batch correction" not in df.columns or df["Batch correction"].isna().all():
        print("No batch-correction scores available (single batch) - skipping tradeoff scatter.")
        return

    x = df["Bio conservation"].astype(float)
    y = df["Batch correction"].astype(float)
    sorted_idx = x.sort_values().index
    x_sorted, y_sorted = x.loc[sorted_idx], y.loc[sorted_idx]

    plt.figure(figsize=(10, 6))
    for method in x_sorted.index:
        color = colors.get(method)
        plt.scatter(x_sorted[method], y_sorted[method], color=color, s=100)
        plt.text(
            x_sorted[method] + 0.001,
            y_sorted[method] + 0.001,
            method,
            fontsize=8,
            ha="left",
            va="bottom",
            color=color,
        )

    plt.xlabel("Bio conservation")
    plt.ylabel("Batch correction")
    plt.title("Tradeoff between Bio Conservation and Batch Correction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path_no_ext}.png", dpi=1000, bbox_inches="tight")
    plt.savefig(f"{save_path_no_ext}.svg", dpi=1000, bbox_inches="tight")
    plt.close()


def _plot_region_vs_celltype_scatter(df_DE, scib_df, colors, save_path_no_ext):
    """df_DE holds per-cell-type conservation (rows=cell types) of the NICHE/region
    label within each cell type - mean±std across cell types is "region conservation".
    scib_df's "Bio conservation" column is the single aggregate cell-type-preservation
    score per method. Mirrors brain/Figure_2_BC.ipynb's region_celltype_scatter cell.
    """
    region_mean = df_DE.mean(axis=0)
    region_std = df_DE.std(axis=0)

    cell_type_cons = scib_df.drop(index="Metric Type", errors="ignore")["Bio conservation"].astype(float)

    common_methods = region_mean.index.intersection(cell_type_cons.index)

    x = region_mean.loc[common_methods]
    x_err = region_std.loc[common_methods]
    y = cell_type_cons.loc[common_methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in common_methods:
        color = colors.get(method)
        ax.errorbar(
            x[method],
            y[method],
            xerr=x_err[method],
            fmt="o",
            color=color,
            label=method,
            markersize=8,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.8,
        )

    ax.set_xlabel("Region Conservation (mean ± std over cell types)", fontsize=11)
    ax.set_ylabel("Cell Type Preservation (Bio Conservation)", fontsize=11)
    ax.set_title("Tradeoff: Region Conservation vs Cell Type Preservation", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_path_no_ext}.png", dpi=500, bbox_inches="tight")
    plt.savefig(f"{save_path_no_ext}.svg", dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal scVI/nicheVI + scib benchmarking runner.")
    parser.add_argument(
        "dataset",
        help="Dataset directory containing params.py, relative to repo root or absolute (e.g. 'brain', 'brain/xenium_ms', 'crc_visium_HD').",
    )
    args = parser.parse_args()

    setup, niche_setup = load_params(args.dataset)
    main(setup, niche_setup)
