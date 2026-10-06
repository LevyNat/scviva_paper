"""
Memory sweep: measure peak GPU and CPU memory for NicheVI training
across different numbers of HVGs and cells.

Sweeps:
  - Gene sweep: 1k, 2k, 3k, 4k HVGs (all cells)
  - Cell sweep: 50k, 100k, 150k, 200k cells (4k HVGs)

Outputs a JSON results file + two plots (genes vs memory, cells vs memory).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import json
import tracemalloc

import numpy as np
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import scvi
import nichevi

# ── paths ──────────────────────────────────────────────────────────────
DATA_FOLDER = "/home/nathanl/Data/VisiumHD_CRC/"
FIGURES_FOLDER = "/home/nathanl/scviva_paper/crc_visium_hd/figures/memory/"
RESULTS_FILE = os.path.join(FIGURES_FOLDER, "nichevi_memory_sweep_results.json")
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# ── data files for gene sweep ──────────────────────────────────────────
HVG_FILES = {
    1000: "adata_legacy_hvg1k.h5ad",
    2000: "adata_legacy_hvg2k.h5ad",
    3000: "adata_legacy_hvg3k.h5ad",
    4000: "adata_legacy_hvg4k.h5ad",
}

# ── cell counts for cell sweep (using 4k HVGs) ────────────────────────
CELL_COUNTS = [50_000, 100_000, 150_000, 200_000]
CELL_SWEEP_FILE = "adata_legacy_hvg4k.h5ad"

# ── config (mirrors params.py) ────────────────────────────────────────
BATCH_KEY = "sample"
SAMPLE_KEY = "sample"
CELL_TYPE_KEY = "cell_type_coarse"
SPATIAL_KEY = "spatial"
K_NN = 20
N_EPOCHS_SCVI = 2
N_EPOCHS_NICHEVI = 2
BATCH_SIZE_SCVI = 512
BATCH_SIZE_NICHEVI = 1024
SEED = 34


def train_and_measure(adata):
    """Train scVI + NicheVI on `adata`, return (peak_gpu_gb, peak_cpu_gb).

    Measures peak GPU/CPU memory across the full pipeline:
    scVI training → latent extraction → NicheVI preprocessing → NicheVI training.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()

    # Filter out cells with very low counts to prevent NaN during training
    counts = adata.layers["counts"]
    if hasattr(counts, "toarray"):
        cell_counts = counts.toarray().sum(axis=1)
    else:
        cell_counts = counts.sum(axis=1)

    n_cells_before = adata.n_obs
    min_counts = 10
    adata = adata[cell_counts >= min_counts].copy()
    print(f"Filtered {n_cells_before - adata.n_obs} cells with < {min_counts} counts")
    print(adata)

    # ── scVI (SCANVI) ─────────────────────────────────────────────────
    scvi.model.SCANVI.setup_anndata(
        adata,
        layer="counts",
        unlabeled_category="ignore",
        batch_key=BATCH_KEY,
        labels_key=CELL_TYPE_KEY,
    )
    scvivae = scvi.model.SCANVI(
        adata,
        gene_likelihood="poisson",
        n_layers=1,
        n_latent=10,
        linear_classifier=True,
    )
    scvivae.train(
        max_epochs=N_EPOCHS_SCVI,
        train_size=0.8,
        validation_size=0.2,
        batch_size=BATCH_SIZE_SCVI,
        plan_kwargs=dict(lr=1e-4, weight_decay=1e-6, optimizer="Adam"),
        early_stopping=False,
    )
    adata.obsm["qz1_m"] = scvivae.get_latent_representation(batch_size=BATCH_SIZE_SCVI)

    # ── NicheVI ───────────────────────────────────────────────────────
    setup_kwargs = {
        "sample_key": SAMPLE_KEY,
        "labels_key": CELL_TYPE_KEY,
        "cell_coordinates_key": SPATIAL_KEY,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(adata, k_nn=K_NN, **setup_kwargs)
    nichevi.nicheSCVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY, **setup_kwargs)

    nichevae = nichevi.nicheSCVI(
        adata,
        cell_rec_weight=1,
        latent_kl_weight=1.0,
        spatial_weight=10,
        niche_rec_weight=10,
        compo_rec_weight=10,
        niche_likelihood="gaussian",
        gene_likelihood="poisson",
        n_layers=1,
        n_layers_niche=1,
        n_layers_compo=1,
        n_hidden_niche=128,
        n_hidden_compo=128,
        n_latent=10,
        use_batch_norm="none",
        use_layer_norm="both",
        prior_mixture=False,
        semisupervised=True,
        linear_classifier=True,
    )
    nichevae.train(
        max_epochs=N_EPOCHS_NICHEVI,
        train_size=0.8,
        validation_size=0.2,
        early_stopping=False,
        batch_size=BATCH_SIZE_NICHEVI,
        plan_kwargs=dict(
            lr=5e-4,
            n_epochs_kl_warmup=40,
            max_kl_weight=1,
            optimizer="Adam",
            weight_decay=1e-6,
            reduce_lr_on_plateau=True,
        ),
    )

    peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    _, peak_cpu_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_cpu_gb = peak_cpu_bytes / 1e9

    del nichevae, scvivae
    gc.collect()
    torch.cuda.empty_cache()

    return peak_gpu_gb, peak_cpu_gb


def plot_results(gene_results, cell_results):
    """Separate GPU and CPU figures, each with gene + cell panels."""
    plt.rcParams["svg.fonttype"] = "none"

    genes = sorted(gene_results.keys())
    cells = sorted(cell_results.keys())

    # ── GPU figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    gpu_gb = [gene_results[g]["peak_gpu_gb"] for g in genes]
    ax.plot(genes, gpu_gb, "o-", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("NicheVI: GPU Memory vs Number of Genes", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gpu_gb = [cell_results[c]["peak_gpu_gb"] for c in cells]
    ax.plot([c / 1000 for c in cells], gpu_gb, "o-", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("Number of Cells (×1000)", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("NicheVI: GPU Memory vs Number of Cells", fontsize=13)
    ax.set_xticks([50, 100, 150, 200])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "nichevi_memory_sweep_gpu.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "nichevi_memory_sweep_gpu.svg"), bbox_inches="tight")
    plt.close()

    # ── CPU figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cpu_gb = [gene_results[g]["peak_cpu_gb"] for g in genes]
    ax.plot(genes, cpu_gb, "s-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("Peak CPU Memory (GB)", fontsize=12)
    ax.set_title("NicheVI: CPU Memory vs Number of Genes", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cpu_gb = [cell_results[c]["peak_cpu_gb"] for c in cells]
    ax.plot([c / 1000 for c in cells], cpu_gb, "s-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Number of Cells (×1000)", fontsize=12)
    ax.set_ylabel("Peak CPU Memory (GB)", fontsize=12)
    ax.set_title("NicheVI: CPU Memory vs Number of Cells", fontsize=13)
    ax.set_xticks([50, 100, 150, 200])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "nichevi_memory_sweep_cpu.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "nichevi_memory_sweep_cpu.svg"), bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {FIGURES_FOLDER}")


def load_checkpoint() -> dict:
    """Load partial results from a previous run, or return empty structure."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        print(f"Loaded existing results from {RESULTS_FILE}")
        data["gene_sweep"] = {int(k): v for k, v in data.get("gene_sweep", {}).items()}
        data["cell_sweep"] = {int(k): v for k, v in data.get("cell_sweep", {}).items()}
        return data
    return {"gene_sweep": {}, "cell_sweep": {}}


def save_checkpoint(all_results: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)


def run_gene_sweep(all_results: dict):
    """Sweep over HVG counts (all cells), resuming from checkpoint if available."""
    results = all_results["gene_sweep"]
    for n_genes, fname in sorted(HVG_FILES.items()):
        if n_genes in results:
            print(f"\nGene sweep: {n_genes} genes — already done, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Gene sweep: {n_genes} genes ({fname})")
        print(f"{'='*60}")

        scvi.settings.seed = SEED
        adata = sc.read_h5ad(os.path.join(DATA_FOLDER, fname))
        n_cells, n_vars = adata.n_obs, adata.n_vars
        print(f"  adata: {n_cells} cells x {n_vars} genes")

        peak_gpu, peak_cpu = train_and_measure(adata)
        print(f"  Peak GPU: {peak_gpu:.2f} GB")
        print(f"  Peak CPU: {peak_cpu:.2f} GB")

        results[n_genes] = {
            "n_cells": n_cells,
            "n_genes": n_vars,
            "peak_gpu_gb": peak_gpu,
            "peak_cpu_gb": peak_cpu,
        }
        save_checkpoint(all_results)

        del adata
        gc.collect()


def run_cell_sweep(all_results: dict):
    """Sweep over cell counts (fixed 4k HVGs), resuming from checkpoint if available."""
    results = all_results["cell_sweep"]
    adata_full = sc.read_h5ad(os.path.join(DATA_FOLDER, CELL_SWEEP_FILE))
    print(f"\nFull dataset: {adata_full.n_obs} cells x {adata_full.n_vars} genes")

    for n_cells in CELL_COUNTS:
        if n_cells in results:
            print(f"\nCell sweep: {n_cells} cells — already done, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Cell sweep: {n_cells} cells ({CELL_SWEEP_FILE})")
        print(f"{'='*60}")
        scvi.settings.seed = SEED

        if n_cells >= adata_full.n_obs:
            adata = adata_full.copy()
        else:
            adata = sc.pp.sample(adata_full, n=n_cells, copy=True, replace=False, rng=SEED)
        print(f"  adata: {adata.n_obs} cells x {adata.n_vars} genes")

        peak_gpu, peak_cpu = train_and_measure(adata)
        print(f"  Peak GPU: {peak_gpu:.2f} GB")
        print(f"  Peak CPU: {peak_cpu:.2f} GB")

        results[n_cells] = {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "peak_gpu_gb": peak_gpu,
            "peak_cpu_gb": peak_cpu,
        }
        save_checkpoint(all_results)

        del adata
        gc.collect()


if __name__ == "__main__":
    print("=" * 60)
    print("NicheVI Memory Sweep")
    print("=" * 60)

    all_results = load_checkpoint()

    run_gene_sweep(all_results)
    run_cell_sweep(all_results)

    print(f"\nResults saved to {RESULTS_FILE}")
    plot_results(all_results["gene_sweep"], all_results["cell_sweep"])
