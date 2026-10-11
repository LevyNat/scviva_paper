"""scVIVA without cell-type labels, for tonsil.

Sweeps several Leiden resolutions, trains one nicheVI per resolution, and scores them all with
scib against the TRUE cell_type so a resolution can be chosen after the fact.

Run:  conda activate scvi && cd slidetags_tonsil/tonsil && python runner_no_labels.py

Differences from the brain version, all forced by this dataset:
  * tonsil has no `adata.raw` (counts already live in layers["counts"]) -- the brain script's
    unconditional `adata.layers["counts"] = adata.raw.X.copy()` would crash.
  * single donor, so scib batch-correction metrics are skipped (they need >1 batch).
  * writes to its own checkpoint folder; the brain script's path would overwrite the existing
    supervised tonsil checkpoint.
  * reuses the stored X_scVI when present (there is no scVI .pt for tonsil, it was trained in
    the notebook) and only trains scVI if it is missing.

Gene-panel experiment (all optional env vars; unset = the original behaviour above):
  NL_DATA_FILE        input file in setup.DATA_FOLDER instead of setup.DATA_FILE. Outputs then
                      go to figures/tonsil_nolabels/<data file name>/ so panels never clobber
                      each other or the original run.
  NL_TARGET_CLUSTERS  "lo,hi": scan the Leiden resolution on a 0.05 grid (on the scVI latent) and
                      take the first value yielding lo..hi clusters; train one nicheVI on it.
  NL_RES_SELECT       "internal": pick the resolution with label-free cluster-quality metrics on the
                      scVI latent -- silhouette (max), Calinski-Harabasz (max), Davies-Bouldin (min)
                      over the 0.05 grid 0.05..2.00; majority vote, silhouette breaks ties. The full
                      sweep is written to leiden_internal_metrics.csv. Overrides NL_TARGET_CLUSTERS.
  NL_TAG              suffix for a variant run on the same data file: outputs go to
                      <data file name>_<tag>_nicheVI.h5ad and figures/tonsil_nolabels/<data file name>_<tag>/
                      (the scVI checkpoint is shared, so variants start from the same scVI).
  NL_REGION           "1": also score region conservation per TRUE cell type (the y-axis of
                      region_celltype_scatter), same procedure as runner.py.
"""

import os
import sys
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from rich import print
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import nichevi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tonsil_10000hvg"))
from params import setup  # same data and model settings as the supervised run (tonsil_10000hvg/params.py)

scvi.settings.seed = 34
plt.rcParams["svg.fonttype"] = "none"
sc.settings._vector_friendly = True
PALETTE = "tab20"

# Leiden resolutions to sweep. 
LEIDEN_RES = [0.15, 0.3, 0.5, 0.6, 0.8, 1.0]

NL_DATA_FILE = os.environ.get("NL_DATA_FILE")
NL_TARGET_CLUSTERS = os.environ.get("NL_TARGET_CLUSTERS")
NL_REGION = os.environ.get("NL_REGION") == "1"
NL_RES_SELECT = os.environ.get("NL_RES_SELECT")
NL_TAG = os.environ.get("NL_TAG")

COMPUTE_UMAP = True
COMPUTE_SCIB = True
SCIB_PROPORTION = 1.0

# own folders, so nothing here can clobber the supervised run
CHECKPOINT_FOLDER = "/home/nathanl/scviva_paper/slidetags_tonsil/checkpoints/tonsil_nolabels"
FIGURES_FOLDER = "/home/nathanl/scviva_paper/slidetags_tonsil/figures/tonsil_nolabels/"

# ----------load data----------#
data_dir, data_file = setup.DATA_FOLDER, NL_DATA_FILE or setup.DATA_FILE
data_file_name = os.path.splitext(data_file)[0]
if NL_DATA_FILE:
    FIGURES_FOLDER = os.path.join(FIGURES_FOLDER, data_file_name + (f"_{NL_TAG}" if NL_TAG else "")) + "/"

path_to_save = os.path.join(CHECKPOINT_FOLDER, data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))
if getattr(setup, "COUNTS_LAYER_FROM_RAW", True):
    adata.layers["counts"] = adata.raw.X.copy()
assert "counts" in adata.layers, "need raw counts in layers['counts']"
print(adata)
print("Will save to:", os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad"))

# ----------scVI (unsupervised -- no labels anywhere)----------#
latent_key = "X_scVI"
scvi_checkpoint = os.path.join(path_to_save, f"scvivae_E{setup.N_EPOCHS_SCVI}_{setup.LIKELIHOOD}.pt")

if latent_key in adata.obsm:
    print(f"[green]reusing existing {latent_key} from the data file[/green]")
elif os.path.exists(scvi_checkpoint):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=setup.BATCH)
    scvivae = scvi.model.SCVI.load(dir_path=scvi_checkpoint, adata=adata)
    adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)
else:
    print("[yellow]training scVI from scratch[/yellow]")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=setup.BATCH)
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
    scvivae.save(dir_path=scvi_checkpoint, save_anndata=False, overwrite=True)
    adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)

# ----------Leiden pseudo-labels----------#
sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15, key_added="scvi_neighbors")

truth = adata.obs[setup.CELL_TYPE].values


def _n_leiden(res):
    sc.tl.leiden(adata, resolution=res, neighbors_key="scvi_neighbors", key_added="_probe",
                 flavor="igraph", n_iterations=2, directed=False)
    return adata.obs["_probe"].nunique()


if NL_RES_SELECT == "internal":
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    _X = np.asarray(adata.obsm[latent_key])
    _rows = []
    for step in range(1, 41):  # 0.05 .. 2.00
        _r = round(0.05 * step, 2)
        k = _n_leiden(_r)
        if k < 2:
            continue
        _lab = adata.obs["_probe"].values
        _rows.append({"resolution": _r, "n_clusters": k,
                      "silhouette": silhouette_score(_X, _lab),
                      "calinski_harabasz": calinski_harabasz_score(_X, _lab),
                      "davies_bouldin": davies_bouldin_score(_X, _lab)})
        print(f"  res={_r:.2f} k={k:2d} sil={_rows[-1]['silhouette']:.3f} "
              f"CH={_rows[-1]['calinski_harabasz']:.0f} DB={_rows[-1]['davies_bouldin']:.3f}")
    _sw = pd.DataFrame(_rows)
    _votes = [_sw.loc[_sw.silhouette.idxmax(), "resolution"],
              _sw.loc[_sw.calinski_harabasz.idxmax(), "resolution"],
              _sw.loc[_sw.davies_bouldin.idxmin(), "resolution"]]
    _vc = pd.Series(_votes).value_counts()
    res = float(_vc.index[0]) if _vc.iloc[0] >= 2 else float(_votes[0])  # majority, else silhouette
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    _sw.to_csv(os.path.join(FIGURES_FOLDER, "leiden_internal_metrics.csv"), index=False)
    LEIDEN_RES = [res]
    del adata.obs["_probe"]
    print(f"[bold]internal-metric votes (sil, CH, DB) = {_votes} -> res {res:.2f} "
          f"({int(_sw.set_index('resolution').loc[res, 'n_clusters'])} clusters)[/bold]")
elif NL_TARGET_CLUSTERS:
    # label-free: the target is a cluster COUNT, never agreement with cell_type. Scan a 0.05 grid
    # upward and take the FIRST resolution that lands in the target range, so the chosen value is
    # a round, reportable number (0.80, not 0.7907) and the choice is reproducible.
    lo_k, hi_k = (int(v) for v in NL_TARGET_CLUSTERS.split(","))
    for step in range(1, 81):  # 0.05 .. 4.00
        res = round(0.05 * step, 2)
        k = _n_leiden(res)
        print(f"  scan res={res:.2f} -> {k} clusters")
        if lo_k <= k <= hi_k:
            break
    else:
        raise RuntimeError(f"no resolution on the 0.05 grid gives {lo_k}-{hi_k} clusters")
    LEIDEN_RES = [res]
    del adata.obs["_probe"]
    print(f"[bold]target {lo_k}-{hi_k} clusters -> res {res:.2f} ({k} clusters)[/bold]")

leiden_summary = []
for res in LEIDEN_RES:
    key = f"leiden_scvi_res{res}"
    sc.tl.leiden(
        adata,
        resolution=res,
        neighbors_key="scvi_neighbors",
        key_added=key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    cl = adata.obs[key].values
    leiden_summary.append(
        {
            "resolution": res,
            "n_clusters": adata.obs[key].nunique(),
            "ARI_vs_cell_type": adjusted_rand_score(truth, cl),
            "NMI_vs_cell_type": normalized_mutual_info_score(truth, cl),
            "min_cluster_size": int(pd.Series(cl).value_counts().min()),
        }
    )
    print(f"res={res}: {adata.obs[key].nunique()} clusters")

leiden_df = pd.DataFrame(leiden_summary)
print(leiden_df.to_string(index=False))
leiden_df.to_csv(os.path.join(FIGURES_FOLDER, "leiden_resolution_summary.csv"), index=False)

if COMPUTE_UMAP:
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(
        adata,
        color=[setup.CELL_TYPE] + [f"leiden_scvi_res{r}" for r in LEIDEN_RES],
        frameon=False,
        ncols=2,
        palette=PALETTE,
        show=False,
    )
    plt.savefig(f"{FIGURES_FOLDER}{latent_key}_umap_nolabel.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{FIGURES_FOLDER}{latent_key}_umap_nolabel.svg", bbox_inches="tight", dpi=300)


# ----------NicheVI per Leiden resolution----------#
def make_niche_config(leiden_res: float) -> dict:
    return {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 10,
        "compo_rec_weight": 10,
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


def setting_name(res):
    return f"{res}_s10_scvi_lr{setup.LR_NICHEVI}_{setup.LIKELIHOOD}"


for res in LEIDEN_RES:
    setting = setting_name(res)
    print("[bold green]" + setting + "[/bold green]")
    setup_dict = make_niche_config(res)

    path_to_save_nichevae = os.path.join(
        path_to_save, f"nichevae_{setting}_{setup.N_EPOCHS_NICHEVI}.pt"
    )

    adata.obsm["qz1_m"] = adata.obsm[latent_key]

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": setup_dict["label_key"],  # <-- Leiden pseudo-labels, not cell_type
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(adata, k_nn=setup.K_NN, **setup_kwargs)
    nichevi.nicheSCVI.setup_anndata(adata, layer="counts", batch_key=setup.BATCH, **setup_kwargs)

    if os.path.exists(path_to_save_nichevae):
        print("Model already exists. Loading...")
        nichevae = nichevi.nicheSCVI.load(dir_path=path_to_save_nichevae, adata=adata)
    else:
        nichevae = nichevi.nicheSCVI(
            adata,
            cell_rec_weight=setup_dict["cell_rec_weight"],
            latent_kl_weight=setup_dict["latent_kl_weight"],
            spatial_weight=setup_dict["spatial_weight"],
            niche_rec_weight=setup_dict["niche_rec_weight"],
            compo_rec_weight=setup_dict["compo_rec_weight"],
            niche_likelihood="gaussian",
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
            early_stopping_patience=100,
        )
        nichevae.save(dir_path=path_to_save_nichevae, save_anndata=False)

    adata.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(
        batch_size=setup.BATCH_SIZE_NICHEVI
    )
    adata.obs[setting + "_compo_error"] = nichevae.get_composition_error(return_mean=False).cpu()
    adata.obs[setting + "_niche_error"] = nichevae.get_niche_error(return_mean=False).cpu()

# ----------save----------#
adata.layers["counts"] = csr_matrix(adata.layers["counts"])
out = os.path.join(path_to_save, data_file_name + (f"_{NL_TAG}" if NL_TAG else "") + "_nicheVI.h5ad")
print("Saving adata to", out)
adata.write_h5ad(out)

# ----------scib: score every resolution against the TRUE cell_type----------#
if COMPUTE_SCIB:
    warnings.filterwarnings("ignore")

    biocons = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )
    # single donor -> batch-correction metrics are undefined, so skip them
    n_batches = adata.obs[setup.BATCH].nunique()
    batchcorr = (
        BatchCorrection(bras=True, ilisi_knn=True, kbet_per_label=True, graph_connectivity=True,
                        pcr_comparison=False)
        if n_batches > 1
        else None
    )
    if batchcorr is None:
        print(f"only {n_batches} batch -- skipping batch-correction metrics")

    embedding_obsm_keys = [latent_key] + [setting_name(r) + "_X_nicheVI" for r in LEIDEN_RES]
    # the supervised counterpart, for reference, if it came along in the data file
    for k in ["s10_scanvi_lr0.0005_poisson_X_nicheVI", "X_scanvi"]:
        if k in adata.obsm:
            embedding_obsm_keys.append(k)

    adata_sampled = (
        adata.copy()
        if SCIB_PROPORTION >= 1.0
        else sc.pp.sample(adata, fraction=SCIB_PROPORTION, copy=True, replace=False, rng=34)
    )

    bm = Benchmarker(
        adata_sampled,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,  # evaluated against the TRUE labels, never trained on them
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)
    plt.savefig(f"{FIGURES_FOLDER}scib_cell_type_nolabel.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{FIGURES_FOLDER}scib_cell_type_nolabel.svg", bbox_inches="tight", dpi=300)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{FIGURES_FOLDER}scib_cell_type_results_nolabel.csv")
    print(scib_df)
    print(f"\nwrote {FIGURES_FOLDER}scib_cell_type_results_nolabel.csv -- pick a resolution from this")

    # ----------region conservation per TRUE cell type (as in runner.py)----------#
    if NL_REGION:
        treshold = getattr(setup, "TRESHOLD", 400)
        adata.obs["cell_type_niche"] = (
            adata.obs[setup.CELL_TYPE].astype(str) + "_" + adata.obs[setup.NICHE].astype(str)
        )
        vc = adata.obs["cell_type_niche"].value_counts()
        sub = adata[adata.obs["cell_type_niche"].isin(vc[vc >= treshold].index)].copy()
        rows = {}
        for ct in sub.obs[setup.CELL_TYPE].unique().tolist():
            sub_ct = sub[sub.obs[setup.CELL_TYPE] == ct].copy()
            if sub_ct.obs["cell_type_niche"].nunique() < 2:
                print(f"Skipping {ct} as it only has one niche")
                continue
            bm_ct = Benchmarker(
                sub_ct,
                batch_key=setup.BATCH,
                label_key="cell_type_niche",
                embedding_obsm_keys=embedding_obsm_keys,
                bio_conservation_metrics=biocons,
                batch_correction_metrics=None,
                n_jobs=-1,
            )
            bm_ct.benchmark()
            rows[ct] = bm_ct.get_results(min_max_scale=False)["Bio conservation"]
        df_region = pd.DataFrame(rows).T.drop(columns="Metric Type", errors="ignore")
        df_region.index.name = "cell_type"
        df_region.to_csv(f"{FIGURES_FOLDER}BioCons_cell_type_nolabel.csv")
        print(df_region.astype(float).agg(["mean", "std"]).T)
