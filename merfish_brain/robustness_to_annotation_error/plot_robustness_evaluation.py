"""
Robustness evaluation for f0-f10: scIB scores + loss diagnostics.

For each flip rate (f0, f1, f5, f10) and seed (34, 42, 123):
  - scIB Bio conservation on validation cells (per-seed val split)
  - For each loss metric: min validation loss + train loss at that same epoch

Loss metrics:
  - Niche composition  (niche_compo_validation / niche_compo_train)
  - Niche reconstruction (niche_reconst_validation / niche_reconst_train)
  - Reconstruction     (reconstruction_loss_validation / reconstruction_loss_train)
  - Classification     (validation_classification_loss / train_classification_loss)

Produces (saved to SAVE_DIR):
  - scib_robustness_f0f10.png  — Bio conservation mean ± std across seeds
  - loss_robustness_f0f10.png  — 4 subplots, val min + train @ val min epoch
  - overfit_gap_f0f10.png      — overfitting gap per metric
"""

import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
SEEDS = [34, 42, 123]
FLIP_NAMES = ["f0", "f1", "f5", "f10"]
FLIP_RATE_MAP = {"f0": 0.0, "f1": 0.01, "f5": 0.05, "f10": 0.10}
FLIP_LABELS = ["f0 (0%)", "f1 (1%)", "f5 (5%)", "f10 (10%)"]

CKPT_DIR = "/home/nathanl/scviva_paper/merfish_brain/checkpoints/adata_M1_M2_core_6_sections_seed"
SCIB_DIR = "/home/nathanl/scviva_paper/merfish_brain/figures/robustness_to_annotation_error/seeds/"
SAVE_DIR = "/home/nathanl/scviva_paper/merfish_brain/figures/robustness_to_annotation_error/seeds/final/"
os.makedirs(SAVE_DIR, exist_ok=True)

# History key pairs: (val_key, train_key, display_label)
LOSS_METRICS = [
    ("niche_compo_validation", "niche_compo_train", "Niche composition"),
    ("niche_reconst_validation", "niche_reconst_train", "Niche reconstruction"),
    ("reconstruction_loss_validation", "reconstruction_loss_train", "Reconstruction"),
    ("validation_classification_loss", "train_classification_loss", "Classification"),
]

SEED_COLORS = {34: "#e41a1c", 42: "#377eb8", 123: "#4daf4a"}


# ------------------------------------------------------------------ #
# Helper: parse history from checkpoint
# ------------------------------------------------------------------ #
def load_history(flip_name: str, seed: int) -> dict | None:
    flip_rate = FLIP_RATE_MAP[flip_name]
    fname = f"nichevae_{flip_name}_s10_scanvi_lr0.0005_poisson_seed{seed}" f"_E1000_seed{seed}_flip{flip_rate}.pt"
    path = os.path.join(CKPT_DIR, fname, "model.pt")
    if not os.path.exists(path):
        print(f"  [WARN] not found: {path}")
        return None
    ckpt = torch.load(path, weights_only=False)
    return ckpt["attr_dict"]["history_"]


def best_val_stats(history: dict, val_key: str, train_key: str) -> tuple[float, float]:
    """
    Returns (min_val_loss, train_loss_at_best_val_epoch).
    """
    val_df = history[val_key]  # DataFrame indexed by epoch
    train_df = history[train_key]

    best_epoch = val_df.iloc[:, 0].idxmin()
    min_val = float(val_df.iloc[:, 0].min())

    # match train epoch: find closest index
    train_vals = train_df.iloc[:, 0]
    if best_epoch in train_vals.index:
        train_at_best = float(train_vals.loc[best_epoch])
    else:
        # pick nearest epoch
        nearest = train_vals.index[np.argmin(np.abs(train_vals.index - best_epoch))]
        train_at_best = float(train_vals.loc[nearest])

    return min_val, train_at_best


# ------------------------------------------------------------------ #
# Collect loss data
# ------------------------------------------------------------------ #
loss_records = []  # {flip_name, seed, metric_label, min_val, train_at_best}

print("Loading checkpoint histories...")
for flip_name in FLIP_NAMES:
    for seed in SEEDS:
        hist = load_history(flip_name, seed)
        if hist is None:
            continue
        for val_key, train_key, label in LOSS_METRICS:
            try:
                min_val, train_at_best = best_val_stats(hist, val_key, train_key)
                loss_records.append(
                    {
                        "flip_name": flip_name,
                        "flip_rate": FLIP_RATE_MAP[flip_name],
                        "seed": seed,
                        "metric": label,
                        "min_val": min_val,
                        "train_at_best": train_at_best,
                        "overfit_gap": min_val - train_at_best,
                    }
                )
            except KeyError as e:
                print(f"  [WARN] missing key {e} for {flip_name} seed {seed}")

loss_df = pd.DataFrame(loss_records)

# ------------------------------------------------------------------ #
# Collect scIB data  (per-seed validation CSVs)
# ------------------------------------------------------------------ #
print("Loading scIB results...")


def parse_flip_from_embedding(emb_name: str) -> str | None:
    """Return flip_name from embedding name, or None if not in FLIP_NAMES."""
    m = re.match(r"(f\d+)_s10_scanvi", emb_name)
    if m is None:
        return None
    flip_name = m.group(1)
    return flip_name if flip_name in FLIP_NAMES else None


scib_records = []
for seed in SEEDS:
    csv_path = os.path.join(SCIB_DIR, f"scib_val_results_robustness_seed{seed}.csv")
    if not os.path.exists(csv_path):
        print(f"  [WARN] not found: {csv_path}")
        continue
    df = pd.read_csv(csv_path, index_col=0)
    for emb_name, row in df.iterrows():
        flip_name = parse_flip_from_embedding(str(emb_name))
        if flip_name is None:
            continue
        scib_records.append(
            {
                "flip_name": flip_name,
                "flip_rate": FLIP_RATE_MAP[flip_name],
                "seed": seed,
                "Bio conservation": row.get("Bio conservation", np.nan),
            }
        )

scib_df = pd.DataFrame(scib_records)
if len(scib_df) > 0:
    scib_df["Bio conservation"] = pd.to_numeric(scib_df["Bio conservation"], errors="coerce")

# ------------------------------------------------------------------ #
# Aggregate helpers
# ------------------------------------------------------------------ #
x_pos = np.arange(len(FLIP_NAMES))


def agg_by_flip(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean ± std across seeds, ordered by FLIP_NAMES."""
    g = df.groupby("flip_name")[value_col].agg(mean="mean", std="std").reindex(FLIP_NAMES)
    return g


def scatter_seeds(ax, df: pd.DataFrame, value_col: str, color: str, jitter: float = 0.06):
    """Plot individual seed dots with horizontal jitter."""
    for i, flip_name in enumerate(FLIP_NAMES):
        sub = df[df["flip_name"] == flip_name]
        for j, (_, row) in enumerate(sub.iterrows()):
            jit = (j - 1) * jitter
            ax.scatter(i + jit, row[value_col], color=color, s=30, alpha=0.6, zorder=4)


def plot_mean_std(ax, agg: pd.DataFrame, color: str, label: str, linewidth: float = 2.0):
    means = agg["mean"].values
    stds = agg["std"].fillna(0).values
    ax.plot(x_pos, means, marker="o", markersize=6, linewidth=linewidth, color=color, label=label, zorder=5)
    ax.fill_between(x_pos, means - stds, means + stds, alpha=0.2, color=color)


# ------------------------------------------------------------------ #
# Figure 1: scIB Bio conservation
# ------------------------------------------------------------------ #
if len(scib_df) > 0:
    BIO_COLOR = "#2ca02c"

    fig, ax = plt.subplots(figsize=(5, 4.5))

    sub = scib_df[["flip_name", "seed", "Bio conservation"]].dropna(subset=["Bio conservation"])
    agg = agg_by_flip(sub, "Bio conservation")
    scatter_seeds(ax, sub, "Bio conservation", color=BIO_COLOR)
    plot_mean_std(ax, agg, color=BIO_COLOR, label="mean ± std")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FLIP_LABELS, fontsize=10)
    ax.set_title("Bio conservation", fontsize=12)
    ax.set_xlabel("Flip rate", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.64, 0.69)
    ax.set_yticks(np.arange(0.64, 0.691, 0.01))
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle("NicheVI — Bio conservation vs. flip rate\n(validation cells, f0–f10, 3 seeds)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "scib_robustness_f0f10.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
else:
    print("[WARN] No scIB data found, skipping Bio conservation figure.")

# ------------------------------------------------------------------ #
# Figure 2: Loss diagnostics
# ------------------------------------------------------------------ #
if len(loss_df) > 0:
    VAL_COLOR = "#d62728"  # red  — min val loss
    TRAIN_COLOR = "#1f77b4"  # blue — train loss at best val epoch

    fig, axes = plt.subplots(1, len(LOSS_METRICS), figsize=(5 * len(LOSS_METRICS), 4.5))

    for ax, (_, _, label) in zip(axes, LOSS_METRICS):
        sub = loss_df[loss_df["metric"] == label]

        agg_val = agg_by_flip(sub, "min_val")
        agg_train = agg_by_flip(sub, "train_at_best")

        # Mean ± std lines
        plot_mean_std(ax, agg_val, VAL_COLOR, label="Val (min)")
        plot_mean_std(ax, agg_train, TRAIN_COLOR, label="Train @ best val epoch")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(FLIP_LABELS, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Flip rate", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "NicheVI — Validation (min) vs. Train loss at best val epoch\n(f0–f10, 3 seeds)",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "loss_robustness_f0f10.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    # Figure 3: Overfitting gap (val_min − train_at_best) per metric
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, len(LOSS_METRICS), figsize=(5 * len(LOSS_METRICS), 4.5))
    GAP_COLOR = "#9467bd"

    for ax, (_, _, label) in zip(axes, LOSS_METRICS):
        sub = loss_df[loss_df["metric"] == label]
        agg = agg_by_flip(sub, "overfit_gap")

        plot_mean_std(ax, agg, GAP_COLOR, label="mean ± std")

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(FLIP_LABELS, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Flip rate", fontsize=9)
        ax.set_ylabel("Val − Train gap", fontsize=9)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "NicheVI — Overfitting gap (val min − train at best val epoch)\n(f0–f10, 3 seeds)",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "overfit_gap_f0f10.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
else:
    print("[WARN] No loss data found, skipping loss/overfitting figures.")

print("\nAll done.")
