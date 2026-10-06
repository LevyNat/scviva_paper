"""Benchmark BANKSY peak CPU memory as a function of number of HVGs and cells.

Uses resource.getrusage (maxrss) — captures all memory including C extensions.
Runs each configuration in a subprocess to isolate peak measurement per run.
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = "/home/nathanl/Data/VisiumHD_CRC/"
# FIGURES_FOLDER = "figures/memory/"
FIGURES_FOLDER = "/home/nathanl/scviva_paper/crc_visium_hd/figures/memory"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

HVG_FILES = {
    1000: "adata_legacy_hvg1k.h5ad",
    2000: "adata_legacy_hvg2k.h5ad",
    3000: "adata_legacy_hvg3k.h5ad",
    4000: "adata_legacy_hvg4k.h5ad",
    # 5000: "adata_legacy_hvg5k.h5ad",
}

# Cell counts to subsample to (None = use all cells)
CELL_COUNTS = [50_000, 100_000, 150_000, 200_000]

# Worker accepts: filepath [n_cells]
WORKER_SCRIPT = r"""
import os, sys, time, json, resource, random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc
sc.settings.verbosity = 0

def run_banksy_core(adata):
    from banksy.main import median_dist_to_nearest_neighbour
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.main import concatenate_all
    from banksy_utils.umap_pca import pca_umap

    adata.var_names_make_unique()
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    k_geom = 20
    max_m = 1
    nbr_weight_decay = "scaled_gaussian"
    lambda_list = [0.2, 0.6]
    pca_dims = [20]

    median_dist_to_nearest_neighbour(adata, key="spatial")
    banksy_dict = initialize_banksy(
        adata, "spatial", k_geom,
        nbr_weight_decay=nbr_weight_decay, max_m=max_m,
        plt_edge_hist=False, plt_nbr_weights=False,
        plt_agf_angles=False, plt_theta=False,
    )
    banksy_dict, banksy_matrix = generate_banksy_matrix(adata, banksy_dict, lambda_list, max_m)
    banksy_dict["nonspatial"] = {0.0: {"adata": concatenate_all([adata.X], 0, adata=adata)}}
    pca_umap(adata, banksy_dict, pca_dims=pca_dims, add_umap=False,
             plt_remaining_var=False, type_to_shuffle=None)

filepath = sys.argv[1]
n_cells = int(sys.argv[2]) if len(sys.argv) > 2 else None

adata = sc.read_h5ad(filepath)

if n_cells is not None and n_cells < adata.n_obs:
    np.random.seed(42)
    idx = np.random.choice(adata.n_obs, n_cells, replace=False)
    idx.sort()
    adata = adata[idx].copy()

n_obs, n_vars = adata.shape

start = time.time()
run_banksy_core(adata)
elapsed = time.time() - start

# maxrss is in KiB on Linux; convert to bytes then to GB (base 10)
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_gb = peak_kb * 1024 / 1e9

print(json.dumps({"peak_memory_GB": peak_gb, "time_s": elapsed, "n_obs": n_obs, "n_vars": n_vars}))
"""


def run_one(filepath, n_cells=None):
    """Run a single BANKSY benchmark in a subprocess."""
    cmd = [sys.executable, "-c", WORKER_SCRIPT, filepath]
    if n_cells is not None:
        cmd.append(str(n_cells))

    # run from this folder so the worker imports the vendored BANKSY (baselines/BANKSY/banksy),
    # as the original did from ~/banksy; the two copies are byte-identical
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))

    if proc.returncode != 0:
        print(f"  ERROR: {proc.stderr[-500:]}")
        return None

    output_lines = proc.stdout.strip().split("\n")
    return json.loads(output_lines[-1])


def plot_results(npz_path=os.path.join(FIGURES_FOLDER, "banksy_memory_benchmark.npz")):
    """Plot from saved .npz file. Can be called independently."""
    plt.rcParams["svg.fonttype"] = "none"
    data = np.load(npz_path)
    genes = data["sweep_genes"]
    gmem = data["sweep_genes_memory"]
    gtime = data["sweep_genes_time"]
    cells = data["sweep_cells"]
    cmem = data["sweep_cells_memory"]
    ctime = data["sweep_cells_time"]

    # ── Memory figure ───────────────────────────────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(genes, gmem, "o-", color="tab:red", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of HVGs", fontsize=12)
    axes[0].set_ylabel("Peak Memory (GB)", fontsize=12)
    axes[0].set_title("BANKSY: CPU Memory vs Number of Genes", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cells / 1000, cmem, "o-", color="tab:red", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Cells (×1000)", fontsize=12)
    axes[1].set_ylabel("Peak Memory (GB)", fontsize=12)
    axes[1].set_title("BANKSY: CPU Memory vs Number of Cells", fontsize=13)
    axes[1].set_xticks([50, 100, 150, 200])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "banksy_memory_sweep_cpu.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "banksy_memory_sweep_cpu.svg"), bbox_inches="tight")
    plt.close()

    # ── Runtime figure ──────────────────────────────────────────────────
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(genes, gtime, "s-", color="tab:orange", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of HVGs", fontsize=12)
    axes[0].set_ylabel("Wall-clock Time (s)", fontsize=12)
    axes[0].set_title("BANKSY: Runtime vs Number of Genes", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cells / 1000, ctime, "s-", color="tab:orange", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Cells (×1000)", fontsize=12)
    axes[1].set_ylabel("Wall-clock Time (s)", fontsize=12)
    axes[1].set_title("BANKSY: Runtime vs Number of Cells", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, "banksy_runtime_sweep.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_FOLDER, "banksy_runtime_sweep.svg"), bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to {FIGURES_FOLDER}")


if __name__ == "__main__":
    # --- Sweep over genes (fixed cells = all) ---
    print("=" * 60)
    print("SWEEP 1: Varying genes (all cells)")
    print("=" * 60)
    gene_results = {}

    for n_genes in sorted(HVG_FILES.keys()):
        filepath = os.path.join(DATA_FOLDER, HVG_FILES[n_genes])
        print(f"\n--- {n_genes} HVGs ---")
        result = run_one(filepath)
        if result:
            gene_results[n_genes] = result
            print(f"  {result['n_obs']} cells x {result['n_vars']} genes")
            print(f"  Peak memory: {result['peak_memory_GB']:.1f} GB | Time: {result['time_s']:.1f}s")

    # --- Sweep over cells (fixed genes = 4k) ---
    print("\n" + "=" * 60)
    print("SWEEP 2: Varying cells (4k HVGs)")
    print("=" * 60)
    cell_results = {}
    filepath_4k = os.path.join(DATA_FOLDER, HVG_FILES[4000])

    for n_cells in CELL_COUNTS:
        print(f"\n--- {n_cells:,} cells ---")
        result = run_one(filepath_4k, n_cells=n_cells)
        if result:
            cell_results[n_cells] = result
            print(f"  {result['n_obs']} cells x {result['n_vars']} genes")
            print(f"  Peak memory: {result['peak_memory_GB']:.1f} GB | Time: {result['time_s']:.1f}s")

    # --- Summary tables ---
    print("\n\n=== SUMMARY: Genes sweep ===")
    print(f"{'HVGs':>6} | {'Peak Mem (GB)':>15} | {'Time (s)':>10} | {'Cells':>8}")
    print("-" * 50)
    for ng in sorted(gene_results):
        r = gene_results[ng]
        print(f"{ng:>6} | {r['peak_memory_GB']:>15.1f} | {r['time_s']:>10.1f} | {r['n_obs']:>8}")

    print("\n=== SUMMARY: Cells sweep ===")
    print(f"{'Cells':>8} | {'Peak Mem (GB)':>15} | {'Time (s)':>10} | {'Genes':>6}")
    print("-" * 50)
    for nc in sorted(cell_results):
        r = cell_results[nc]
        print(f"{nc:>8} | {r['peak_memory_GB']:>15.1f} | {r['time_s']:>10.1f} | {r['n_vars']:>6}")

    # --- Save results ---
    np.savez(
        os.path.join(FIGURES_FOLDER, "banksy_memory_benchmark.npz"),
        # Gene sweep
        sweep_genes=np.array(sorted(gene_results.keys())),
        sweep_genes_memory=np.array([gene_results[k]["peak_memory_GB"] for k in sorted(gene_results)]),
        sweep_genes_time=np.array([gene_results[k]["time_s"] for k in sorted(gene_results)]),
        # Cell sweep
        sweep_cells=np.array(sorted(cell_results.keys())),
        sweep_cells_memory=np.array([cell_results[k]["peak_memory_GB"] for k in sorted(cell_results)]),
        sweep_cells_time=np.array([cell_results[k]["time_s"] for k in sorted(cell_results)]),
    )

    plot_results()
