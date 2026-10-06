"""
Plot per-cell-type Bio conservation scores across flip rates with error bars.

Loads BioCons_cell_type_robustness_val_seed{N}.csv for seeds 34, 42, 123,
computes mean ± std across seeds for each (cell_type, flip_rate) pair,
and produces:
  1. Multi-line plot: all cell types × flip rates, error bands, one panel
  2. Individual cell-type panels: score vs flip rate with shaded std

Usage:
    python plot_cell_type_robustness.py
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams["svg.fonttype"] = "none"

SEEDS = [34, 42, 123]
DATA_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/seeds/"
SAVE_DIR = DATA_DIR  # save alongside the CSVs

FLIP_RATE_ORDER = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 1.0]
FLIP_LABEL_MAP = {
    0.0: "f0\n(0%)",
    0.01: "f1\n(1%)",
    0.05: "f5\n(5%)",
    0.10: "f10\n(10%)",
    0.20: "f20\n(20%)",
    0.40: "f40\n(40%)",
    0.60: "f60\n(60%)",
    1.0: "f100\n(100%)",
}


def parse_flip_rate(col_name: str) -> float:
    """Extract flip rate float from column name ending in _flip{X}."""
    m = re.search(r"_flip([\d.]+)$", col_name)
    if m is None:
        raise ValueError(f"Cannot parse flip rate from: {col_name}")
    return float(m.group(1))


def load_seed(seed: int) -> pd.DataFrame:
    """Load one seed CSV and return long-format DataFrame."""
    path = os.path.join(DATA_DIR, f"BioCons_cell_type_robustness_val_seed{seed}.csv")
    df = pd.read_csv(path, index_col=0)  # rows = cell types, cols = embedding names
    rows = []
    for col in df.columns:
        flip_rate = parse_flip_rate(col)
        for cell_type, score in df[col].items():
            rows.append({"seed": seed, "cell_type": cell_type, "flip_rate": flip_rate, "score": score})
    return pd.DataFrame(rows)


# ---- Load all seeds ----
frames = [load_seed(s) for s in SEEDS]
long_df = pd.concat(frames, ignore_index=True)

# Enforce flip rate order
long_df = long_df[long_df["flip_rate"].isin(FLIP_RATE_ORDER)].copy()
long_df["flip_rate"] = pd.Categorical(long_df["flip_rate"], categories=FLIP_RATE_ORDER, ordered=True)

# Aggregate: mean ± std across seeds
agg = (
    long_df.groupby(["cell_type", "flip_rate"])["score"]
    .agg(mean="mean", std="std")
    .reset_index()
)
agg["flip_rate"] = agg["flip_rate"].astype(float)

cell_types = sorted(agg["cell_type"].unique())
n_ct = len(cell_types)
colors = cm.tab10(np.linspace(0, 1, n_ct))
flip_rates_sorted = FLIP_RATE_ORDER
x_pos = np.arange(len(flip_rates_sorted))

# =========================================================
# Plot 1: All cell types in one figure (multi-line + error band)
# =========================================================
fig, ax = plt.subplots(figsize=(10, 5))

for ct, color in zip(cell_types, colors):
    sub = agg[agg["cell_type"] == ct].sort_values("flip_rate")
    means = sub["mean"].values
    stds = sub["std"].fillna(0).values
    ax.plot(x_pos, means, marker="o", markersize=4, linewidth=1.5, label=ct, color=color)
    ax.fill_between(x_pos, means - stds, means + stds, alpha=0.15, color=color)

ax.set_xticks(x_pos)
ax.set_xticklabels([FLIP_LABEL_MAP[r] for r in flip_rates_sorted], fontsize=9)
ax.set_xlabel("Label flip rate", fontsize=11)
ax.set_ylabel("Bio conservation (scIB)", fontsize=11)
ax.set_title("NicheVI — Bio conservation per cell type vs. flip rate\n(mean ± std, 3 seeds)", fontsize=12)
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()

out_all = os.path.join(SAVE_DIR, "celltype_robustness_all_celltypes.png")
fig.savefig(out_all, dpi=300, bbox_inches="tight")
fig.savefig(out_all.replace(".png", ".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_all}")

# =========================================================
# Plot 2: Individual panels — one subplot per cell type
# =========================================================
n_cols = 3
n_rows = int(np.ceil(n_ct / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

for i, ct in enumerate(cell_types):
    ax = axes[i // n_cols][i % n_cols]
    sub = agg[agg["cell_type"] == ct].sort_values("flip_rate")
    means = sub["mean"].values
    stds = sub["std"].fillna(0).values
    color = colors[i]

    ax.plot(x_pos, means, marker="o", markersize=5, linewidth=2, color=color)
    ax.fill_between(x_pos, means - stds, means + stds, alpha=0.25, color=color)
    ax.set_title(ct, fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([FLIP_LABEL_MAP[r] for r in flip_rates_sorted], fontsize=7)
    ax.set_ylabel("Bio conservation", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

# Hide unused axes
for j in range(n_ct, n_rows * n_cols):
    axes[j // n_cols][j % n_cols].set_visible(False)

fig.suptitle("NicheVI — Bio conservation per cell type vs. flip rate (mean ± std, 3 seeds)", fontsize=13)
fig.tight_layout()

out_panels = os.path.join(SAVE_DIR, "celltype_robustness_individual_panels.png")
fig.savefig(out_panels, dpi=300, bbox_inches="tight")
fig.savefig(out_panels.replace(".png", ".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_panels}")

# =========================================================
# Plot 3: Heatmap — mean score, rows=cell type, cols=flip rate
# =========================================================
pivot_mean = agg.pivot(index="cell_type", columns="flip_rate", values="mean")
pivot_mean = pivot_mean[flip_rates_sorted]  # enforce column order

fig, ax = plt.subplots(figsize=(10, 0.55 * n_ct + 1.5))
im = ax.imshow(pivot_mean.values, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=0.6)
plt.colorbar(im, ax=ax, label="Bio conservation (scIB)")
ax.set_xticks(np.arange(len(flip_rates_sorted)))
ax.set_xticklabels([FLIP_LABEL_MAP[r] for r in flip_rates_sorted], fontsize=9)
ax.set_yticks(np.arange(n_ct))
ax.set_yticklabels(pivot_mean.index, fontsize=9)
ax.set_xlabel("Label flip rate", fontsize=11)
ax.set_title("NicheVI — Mean Bio conservation per cell type (mean across 3 seeds)", fontsize=12)
fig.tight_layout()

out_hmap = os.path.join(SAVE_DIR, "celltype_robustness_heatmap.png")
fig.savefig(out_hmap, dpi=300, bbox_inches="tight")
fig.savefig(out_hmap.replace(".png", ".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_hmap}")

# =========================================================
# Plot 4: Per-seed raw lines (faint) + mean bold — one per cell type
# =========================================================
seed_colors = {34: "#e41a1c", 42: "#377eb8", 123: "#4daf4a"}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)

for i, ct in enumerate(cell_types):
    ax = axes[i // n_cols][i % n_cols]
    sub_agg = agg[agg["cell_type"] == ct].sort_values("flip_rate")

    # Per-seed faint lines
    for seed in SEEDS:
        raw = long_df[(long_df["cell_type"] == ct) & (long_df["seed"] == seed)].sort_values("flip_rate")
        ax.plot(x_pos, raw["score"].values, marker=".", markersize=4, linewidth=0.8,
                color=seed_colors[seed], alpha=0.5, label=f"seed {seed}")

    # Mean bold line
    means = sub_agg["mean"].values
    stds = sub_agg["std"].fillna(0).values
    ax.plot(x_pos, means, marker="o", markersize=6, linewidth=2.5, color="black", label="mean", zorder=5)
    ax.fill_between(x_pos, means - stds, means + stds, alpha=0.15, color="grey")

    ax.set_title(ct, fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([FLIP_LABEL_MAP[r] for r in flip_rates_sorted], fontsize=7)
    ax.set_ylabel("Bio conservation", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, frameon=False)

# Hide unused
for j in range(n_ct, n_rows * n_cols):
    axes[j // n_cols][j % n_cols].set_visible(False)

fig.suptitle("NicheVI — Bio conservation per cell type, per seed + mean ± std", fontsize=13)
fig.tight_layout()

out_seeds = os.path.join(SAVE_DIR, "celltype_robustness_per_seed.png")
fig.savefig(out_seeds, dpi=300, bbox_inches="tight")
fig.savefig(out_seeds.replace(".png", ".svg"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_seeds}")

print("\nDone!")
