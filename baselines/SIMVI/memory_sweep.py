"""
Memory sweep: measure peak GPU and CPU memory for SIMVI training
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
import re
import tracemalloc

import numpy as np
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from simvi.model import SimVI

# ── paths ──────────────────────────────────────────────────────────────
DATA_FOLDER = "/home/nathanl/Data/VisiumHD_CRC/"
CHECKPOINT_FOLDER = "/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/"
FIGURES_FOLDER = "/home/nathanl/scviva_paper/crc_visium_hd/figures/memory/"
RESULTS_FILE = os.path.join(FIGURES_FOLDER, "memory_sweep_results.json")
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

# ── SIMVI / training config (matches simvi_runner_2025.py) except for #epochs ────────────
BATCH_KEY = "sample"
SAMPLE_KEY = "sample"
SPATIAL_KEY = "spatial"
K_NN = 20
N_EPOCHS = 100
MAE_EPOCHS = 50
BATCH_SIZE = 1024
TRAIN_SIZE = 0.8
SEED = 34


def seed_everything(seed: int = 34):
    import random
    import pytorch_lightning as pl

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def train_and_measure(adata):
    """Train SIMVI on `adata` and return (peak_gpu_mb, peak_cpu_mb)."""
    # Reset GPU stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Start CPU tracking
    tracemalloc.start()

    SimVI.setup_anndata(adata, layer="counts", batch_key=BATCH_KEY)
    edge_index = SimVI.extract_edge_index(adata, batch_key=SAMPLE_KEY, spatial_key=SPATIAL_KEY, n_neighbors=K_NN)

    model = SimVI(
        adata,
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

    model.train(
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

    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e9
    _, peak_cpu_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_cpu_mb = peak_cpu_bytes / 1e9

    # Clean up
    del model, edge_index
    gc.collect()
    torch.cuda.empty_cache()

    return peak_gpu_mb, peak_cpu_mb


def load_gpu_from_log(fname: str) -> float | None:
    """Return Max allocated MB from an existing gpu_log file, or None if absent."""
    stem = fname.replace(".h5ad", "")
    log_path = os.path.join(CHECKPOINT_FOLDER, fname, f"gpu_log_{stem}.txt")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    match = re.search(r"Max allocated:\s+([\d.]+)\s+MB", content)
    return float(match.group(1)) / 1e3 if match else None  # MB (base-10) → GB


# Gene counts for which training is skipped and the existing gpu_log is used.
LOG_ONLY_GENES = {4000}


def plot_results(gene_results, cell_results):
    """Separate GPU and CPU figures, each with gene + cell panels."""
    plt.rcParams["svg.fonttype"] = "none"

    genes = sorted(gene_results.keys())
    cells = sorted(cell_results.keys())

    # ── GPU figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    gpu_gb = [gene_results[g]["peak_gpu_mb"] for g in genes]
    ax.plot(genes, gpu_gb, "o-", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("simVI: GPU Memory vs Number of Genes", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gpu_gb = [cell_results[c]["peak_gpu_mb"] for c in cells]
    ax.plot([c / 1000 for c in cells], gpu_gb, "o-", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("Number of Cells (×1000)", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("simVI: GPU Memory vs Number of Cells", fontsize=13)
    ax.set_xticks([50, 100, 150, 200])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "simvi_memory_sweep_gpu.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "simvi_memory_sweep_gpu.svg"), bbox_inches="tight")
    plt.close()

    # ── CPU figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    cpu_gb = [v if (v := gene_results[g]["peak_cpu_mb"]) is not None else float("nan") for g in genes]
    ax.plot(genes, cpu_gb, "s-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Number of Genes", fontsize=12)
    ax.set_ylabel("Peak CPU Memory (GB)", fontsize=12)
    ax.set_title("simVI: CPU Memory vs Number of Genes", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cpu_gb = [cell_results[c]["peak_cpu_mb"] for c in cells]
    ax.plot([c / 1000 for c in cells], cpu_gb, "s-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("Number of Cells (×1000)", fontsize=12)
    ax.set_ylabel("Peak CPU Memory (GB)", fontsize=12)
    ax.set_title("simVI: CPU Memory vs Number of Cells", fontsize=13)
    ax.set_xticks([50, 100, 150, 200])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "simvi_memory_sweep_cpu.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "simvi_memory_sweep_cpu.svg"), bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {FIGURES_FOLDER}")


def load_checkpoint() -> dict:
    """Load partial results from a previous run, or return empty structure."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        print(f"Loaded existing results from {RESULTS_FILE}")
        # JSON keys are strings; convert back to int. Values stored in MB (base-10) → GB.
        def _to_gb(entry):
            return {**entry, **{k: entry[k] / 1e3 for k in ("peak_gpu_mb", "peak_cpu_mb", "peak_gpu_mb_logged") if entry.get(k) is not None}}
        data["gene_sweep"] = {int(k): _to_gb(v) for k, v in data.get("gene_sweep", {}).items()}
        data["cell_sweep"] = {int(k): _to_gb(v) for k, v in data.get("cell_sweep", {}).items()}
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

        logged_gpu = load_gpu_from_log(fname)
        adata = sc.read_h5ad(os.path.join(DATA_FOLDER, fname))
        n_cells, n_vars = adata.n_obs, adata.n_vars
        print(f"  adata: {n_cells} cells x {n_vars} genes")

        if n_genes in LOG_ONLY_GENES:
            if logged_gpu is None:
                raise FileNotFoundError(
                    f"No gpu_log found for {fname} and training was skipped. "
                    "Run training manually or remove {n_genes} from LOG_ONLY_GENES."
                )
            print("  Skipping training — using logged value.")
            print(f"  Peak GPU (logged): {logged_gpu:.2f} GB")
            peak_gpu, peak_cpu = logged_gpu, None
        else:
            seed_everything(SEED)
            peak_gpu, peak_cpu = train_and_measure(adata)
            print(f"  Peak GPU (fresh): {peak_gpu:.2f} GB")
            if logged_gpu is not None:
                diff_pct = abs(peak_gpu - logged_gpu) / logged_gpu * 100
                print(f"  Peak GPU (logged): {logged_gpu:.2f} GB  |  diff: {diff_pct:.1f}%")
            print(f"  Peak CPU: {peak_cpu:.2f} GB")

        results[n_genes] = {
            "n_cells": n_cells,
            "n_genes": n_vars,
            "peak_gpu_mb": peak_gpu,
            "peak_gpu_mb_logged": logged_gpu,
            "peak_cpu_mb": peak_cpu,
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
        seed_everything(SEED)

        if n_cells >= adata_full.n_obs:
            adata = adata_full.copy()
        else:
            adata = sc.pp.sample(adata_full, n=n_cells, copy=True, replace=False, rng=SEED)
        print(f"  adata: {adata.n_obs} cells x {adata.n_vars} genes")

        peak_gpu, peak_cpu = train_and_measure(adata)
        results[n_cells] = {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "peak_gpu_mb": peak_gpu,
            "peak_cpu_mb": peak_cpu,
        }
        save_checkpoint(all_results)
        print(f"  Peak GPU: {peak_gpu:.2f} GB")
        print(f"  Peak CPU: {peak_cpu:.2f} GB")

        del adata
        gc.collect()


if __name__ == "__main__":
    print("=" * 60)
    print("SIMVI Memory Sweep")
    print("=" * 60)

    all_results = load_checkpoint()

    run_gene_sweep(all_results)
    run_cell_sweep(all_results)

    print(f"\nResults saved to {RESULTS_FILE}")
    plot_results(all_results["gene_sweep"], all_results["cell_sweep"])
