import argparse
import sys
import os

# CUDA_VISIBLE_DEVICES must be set before torch (or anything that imports torch, like
# simvi/scanpy) is ever imported - the CUDA driver enumerates/caches visible devices on
# first touch, so setting this any later silently has no effect and the process falls
# back to the default physical GPU 0 regardless of --gpu.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_params import DATASETS, add_dataset_arg

parser = argparse.ArgumentParser()
add_dataset_arg(parser)
parser.add_argument("--gpu", default="1", help="value for CUDA_VISIBLE_DEVICES")
args = parser.parse_args()
setup = DATASETS[args.dataset]

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import scanpy as sc
import numpy as np

# anndata 0.11.4's h5ad reader still references np.string_, removed in
# NumPy 2.0 (https://github.com/scverse/anndata/blob/0.11.4/src/anndata/compat/__init__.py#L130).
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import matplotlib.pyplot as plt
from simvi.model import SimVI
import torch


def seed_everything(seed: int = 34):
    import os
    import random
    import numpy as np
    import torch
    import pytorch_lightning as pl

    """Set seed for reproducibility across torch, numpy, random, and Lightning."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA < 11.2 compatibility

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pl.seed_everything(seed, workers=True)
    print(f"✅ Seed set to {seed} with full deterministic mode.")


seed_everything()

COORDS = setup.COORDS
BATCH = setup.BATCH
SAMPLE = setup.SAMPLE
LABEL = setup.CELL_TYPE
NICHE = setup.NICHE

N_EPOCHS = 200
BATCH_SIZE = 1024
MAE_EPOCHS = 50

K_NN = 20
SHUFFLE_ALL = False

simvi_is_trained = False
scib_proportion = 0.3

TRAIN_SIZE = 0.8
TRAIN_SIZE_STR = str(TRAIN_SIZE).replace(".", "")

folder = f"/simvi25_t{TRAIN_SIZE_STR}_{N_EPOCHS}epochs_mae_{MAE_EPOCHS}"

if SHUFFLE_ALL:
    folder += "_shuffled"


data_dir = setup.DATA_FOLDER
data_file_name = setup.DATA_FILE

FIGURES_FOLDER = setup.FIGURES_FOLDER
os.makedirs(FIGURES_FOLDER, exist_ok=True)

path_to_save = os.path.join(setup.CHECKPOINTS_FOLDER, data_file_name)
os.makedirs(path_to_save, exist_ok=True)
adata = sc.read_h5ad(os.path.join(data_dir, data_file_name))
print(adata)


if adata.raw:
    adata.layers["counts"] = adata.raw.X.copy()

# -----------------------------------
# Optional: shuffle coordinates globally
# -----------------------------------
if SHUFFLE_ALL:
    # -----------------------------------
    # Save original spatial coordinates
    # -----------------------------------
    if "spatial_orig" not in adata.obsm.keys():
        adata.obsm["spatial_orig"] = adata.obsm["spatial"].copy()

    print("[bold red]Shuffling spatial coordinates PER SLIDE![bold red]")

    # Copy the region labels before shuffling
    niche_labels = adata.obs[NICHE].values.copy()

    # Go slide by slide
    for section in adata.obs[SAMPLE].unique():
        mask = adata.obs[SAMPLE] == section

        # Shuffle coordinates within this slide
        coords = adata.obsm["spatial"][mask]
        shuffled_coords = coords[np.random.permutation(coords.shape[0])]
        adata.obsm["spatial"][mask] = shuffled_coords

        # Shuffle region labels within this slide
        region_sub = niche_labels[mask]
        shuffled_region = region_sub[np.random.permutation(len(region_sub))]
        niche_labels[mask] = shuffled_region

    # Update shuffled region labels
    adata.obs[NICHE] = niche_labels


SimVI.setup_anndata(adata, layer="counts", batch_key=BATCH)
edge_index = SimVI.extract_edge_index(adata, batch_key=SAMPLE, spatial_key="spatial", n_neighbors=K_NN)


if not simvi_is_trained:
    model = SimVI(
        adata,
        # n_latent=10,
        # dropout_rate=0.1,
        # kl_weight=1,
        # kl_gatweight=1,
        # lam_mi=5,
        n_hidden=128,
        n_intrinsic=10,
        n_spatial=10,
        n_layers=1,
        dropout_rate=0,
        use_observed_lib_size=True,
        lam_mi=1000,
        reg_to_use="mmd",
        noising_mode="sampling",
        dis_to_use="zinb",
        permutation_rate=0.25,
        var_eps=1e-4,
        kl_weight=1,
        kl_gatweight=0.01,
        attention_heads=1,
    )
    train_loss, val_loss = model.train(
        edge_index,
        max_epochs=N_EPOCHS,
        train_size=TRAIN_SIZE,
        validation_size=1 - TRAIN_SIZE,
        anneal_epochs=50,
        mae_epochs=MAE_EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        use_gpu=True,
        batch_size=BATCH_SIZE,
    )

    # Log GPU memory after quick training
    # import torch

    # gpu_log_file = f"gpu_log_{data_file_name.replace('.h5ad', '')}.txt"
    # path_to_save_gpu_log = os.path.join(path_to_save, gpu_log_file)
    # print(f"Saving GPU log to {path_to_save_gpu_log}")
    # with open(path_to_save_gpu_log, "w") as f:
    #     f.write(f"Quick test for {data_file_name}\n")
    #     f.write(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    #     f.write(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB\n")
    #     f.write(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB\n")
    #     f.write(f"Max reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB\n")

    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.yscale("log")

    plt.savefig(path_to_save + folder + "mae_" + str(MAE_EPOCHS) + "_loss25.png")

    model.save(path_to_save + folder + "mae_" + str(MAE_EPOCHS) + ".pt", overwrite=True)


else:
    model = SimVI.load(
        dir_path=path_to_save + folder + "mae_" + str(MAE_EPOCHS) + ".pt",
        adata=adata,
    )

# -----------------------------------
# Save SimVI representations — name depends on shuffle
# -----------------------------------
suffix = "_shuffled" if SHUFFLE_ALL else ""

adata.obsm[f"simvi25_intrinsic_mae_{MAE_EPOCHS}{suffix}"] = model.get_latent_representation(
    edge_index, representation_kind="intrinsic", give_mean=True
)
adata.obsm[f"simvi25_interact_mae_{MAE_EPOCHS}{suffix}"] = model.get_latent_representation(
    edge_index, representation_kind="interaction", give_mean=True
)
adata.obsm[f"simvi25_all_mae_{MAE_EPOCHS}{suffix}"] = model.get_latent_representation(
    edge_index, representation_kind="all", give_mean=True
)
adata.obsm[f"simvi25_both_mae_{MAE_EPOCHS}{suffix}"] = np.concatenate(
    [
        adata.obsm[f"simvi25_interact_mae_{MAE_EPOCHS}{suffix}"],
        adata.obsm[f"simvi25_intrinsic_mae_{MAE_EPOCHS}{suffix}"],
    ],
    axis=1,
)


print(adata.obsm.keys())

# adata_subset = adata[adata.obs[LABEL].isin(["astrocyte"])].copy()
# adata_subset.write_h5ad(
#     "/home/nathanl/scviva_paper/merfish_brain/duplicate_region/adata_astrocyte_only_simvi.h5ad"
# )

if not SHUFFLE_ALL:
    adata.write_h5ad(os.path.join(data_dir, data_file_name))
    print(f"Saved data to {data_file_name}")
else:
    shuffled_file_name = data_file_name.replace(".h5ad", "_simvi25_shuffled.h5ad")
    adata.write_h5ad(os.path.join(data_dir, shuffled_file_name))
    print(f"Saved data to {shuffled_file_name}")


if SHUFFLE_ALL:
    from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation
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

    embedding_obsm_keys = (
        f"simvi25_intrinsic_mae_{MAE_EPOCHS}{suffix}",
        f"simvi25_interact_mae_{MAE_EPOCHS}{suffix}",
        # f"simvi25_all_mae_{MAE_EPOCHS}{suffix}",
        f"simvi25_both_mae_{MAE_EPOCHS}{suffix}",
    )

    if scib_proportion < 1.0:
        adata_sampled = sc.pp.sample(adata, fraction=scib_proportion, copy=True, replace=False, rng=34)
    else:
        adata_sampled = adata.copy()

    bm = Benchmarker(
        adata_sampled,
        batch_key=BATCH,
        label_key=LABEL,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(f"{FIGURES_FOLDER}scib_cell_type.png", bbox_inches="tight", dpi=800)
    plt.savefig(f"{FIGURES_FOLDER}scib_cell_type.svg", bbox_inches="tight", dpi=800)

    scib_df = bm.get_results(min_max_scale=False, clean_names=True)
    scib_df.to_csv(f"{FIGURES_FOLDER}scib_cell_type_results.csv")
