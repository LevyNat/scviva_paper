"""
Robustness evaluation for f0-f10: scIB scores + loss diagnostics.
NicheVI and scANVI compared.

Loss metrics:
  NicheVI:  Niche composition, Niche reconstruction, Reconstruction, Classification
  scANVI:   Reconstruction, Classification  (no niche losses)

Produces:
  - scib_robustness_f0f10.png       — Bio conservation, NicheVI vs scANVI
  - loss_robustness_f0f10.png       — 4 subplots, NicheVI val min + train @ val min epoch
  - loss_robustness_scanvi_f0f10.png — 2 subplots, scANVI (reconstruction + classification)
  - overfit_gap_f0f10.png           — NicheVI overfitting gap
  - overfit_gap_scanvi_f0f10.png    — scANVI overfitting gap
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

CKPT_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/adata_legacy_hvg4k_robustness"
SCIB_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/seeds/"
SAVE_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/seeds/journal/"
os.makedirs(SAVE_DIR, exist_ok=True)

# NicheVI loss metrics
NICHEVI_LOSS_METRICS = [
    ("niche_compo_validation", "niche_compo_train", "Niche composition"),
    ("niche_reconst_validation", "niche_reconst_train", "Niche reconstruction"),
    ("reconstruction_loss_validation", "reconstruction_loss_train", "Reconstruction"),
    ("validation_classification_loss", "train_classification_loss", "Classification"),
]

# scANVI loss metrics (subset that exists in scANVI checkpoints)
SCANVI_LOSS_METRICS = [
    ("reconstruction_loss_validation", "reconstruction_loss_train", "Reconstruction"),
    ("validation_classification_loss", "train_classification_loss", "Classification"),
]

# Model colours
NICHEVI_COLOR = "#2ca02c"  # green
SCANVI_COLOR = "#1f77b4"  # blue
VAL_COLOR = "#d62728"  # red   — min val loss
TRAIN_COLOR = "#ff7f0e"  # orange — train @ best val epoch
GAP_COLOR = "#9467bd"  # purple

VAL_COLOR = "#d62728"  # red  — min val loss
TRAIN_COLOR = "#1f77b4"  # blue — train loss at best val epoch


# ------------------------------------------------------------------ #
# Checkpoint loaders
# ------------------------------------------------------------------ #
def load_history_nichevi(flip_name: str, seed: int) -> dict | None:
    flip_rate = FLIP_RATE_MAP[flip_name]
    fname = f"nichevae_{flip_name}_s10_scanvi_lr0.0005_poisson_seed{seed}_E1000" f"_seed{seed}_flip{flip_rate}.pt"
    path = os.path.join(CKPT_DIR, fname, "model.pt")
    if not os.path.exists(path):
        print(f"  [WARN] NicheVI not found: {path}")
        return None
    ckpt = torch.load(path, weights_only=False)
    return ckpt["attr_dict"]["history_"]


def load_history_scanvi(flip_name: str, seed: int) -> dict | None:
    flip_rate = FLIP_RATE_MAP[flip_name]
    fname = f"scanvivae_E1000_poisson_seed{seed}_flip{flip_rate}.pt"
    path = os.path.join(CKPT_DIR, fname, "model.pt")
    if not os.path.exists(path):
        print(f"  [WARN] scANVI not found: {path}")
        return None
    ckpt = torch.load(path, weights_only=False)
    return ckpt["attr_dict"]["history_"]


def best_val_stats(history: dict, val_key: str, train_key: str) -> tuple[float, float]:
    val_df = history[val_key]
    train_df = history[train_key]
    best_epoch = val_df.iloc[:, 0].idxmin()
    min_val = float(val_df.iloc[:, 0].min())
    train_vals = train_df.iloc[:, 0]
    if best_epoch in train_vals.index:
        train_at_best = float(train_vals.loc[best_epoch])
    else:
        nearest = train_vals.index[np.argmin(np.abs(train_vals.index - best_epoch))]
        train_at_best = float(train_vals.loc[nearest])
    return min_val, train_at_best


def collect_loss_records(loader_fn, metrics):
    records = []
    for flip_name in FLIP_NAMES:
        for seed in SEEDS:
            hist = loader_fn(flip_name, seed)
            if hist is None:
                continue
            for val_key, train_key, label in metrics:
                try:
                    min_val, train_at_best = best_val_stats(hist, val_key, train_key)
                    records.append(
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
    return pd.DataFrame(records)


# ------------------------------------------------------------------ #
# Load loss histories
# ------------------------------------------------------------------ #
print("Loading NicheVI checkpoint histories...")
nichevi_loss_df = collect_loss_records(load_history_nichevi, NICHEVI_LOSS_METRICS)

print("Loading scANVI checkpoint histories...")
scanvi_loss_df = collect_loss_records(load_history_scanvi, SCANVI_LOSS_METRICS)

# ------------------------------------------------------------------ #
# Load scIB data  (per-seed validation CSVs)
# ------------------------------------------------------------------ #
print("Loading scIB results...")


def load_scib_val(csv_template, flip_name_parser):
    records = []
    for seed in SEEDS:
        csv_path = os.path.join(SCIB_DIR, csv_template.format(seed=seed))
        df = pd.read_csv(csv_path, index_col=0)
        for emb_name, row in df.iterrows():
            flip_name = flip_name_parser(str(emb_name))
            if flip_name is None:
                continue
            records.append(
                {
                    "flip_name": flip_name,
                    "flip_rate": FLIP_RATE_MAP[flip_name],
                    "seed": seed,
                    "Bio conservation": row.get("Bio conservation", np.nan),
                }
            )
    out = pd.DataFrame(records)
    out["Bio conservation"] = pd.to_numeric(out["Bio conservation"], errors="coerce")
    return out


def parse_flip_nichevi(emb_name):
    m = re.match(r"(f\d+)_s10_scanvi", emb_name)
    if m is None:
        return None
    fn = m.group(1)
    return fn if fn in FLIP_NAMES else None


def parse_flip_scanvi(emb_name):
    m = re.search(r"_flip([\d.]+)$", emb_name)
    if m is None:
        return None
    rate = float(m.group(1))
    inv = {v: k for k, v in FLIP_RATE_MAP.items()}
    return inv.get(rate)


nichevi_scib_df = load_scib_val("scib_val_results_robustness_seed{seed}.csv", parse_flip_nichevi)
scanvi_scib_df = load_scib_val("scib_val_results_robustness_scanvi_seed{seed}.csv", parse_flip_scanvi)

# ------------------------------------------------------------------ #
# Plot helpers
# ------------------------------------------------------------------ #
x_pos = np.arange(len(FLIP_NAMES))


def agg_by_flip(df, value_col):
    return df.groupby("flip_name")[value_col].agg(mean="mean", std="std").reindex(FLIP_NAMES)


def scatter_seeds(ax, df, value_col, color, jitter=0.06):
    for i, flip_name in enumerate(FLIP_NAMES):
        sub = df[df["flip_name"] == flip_name]
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(i + (j - 1) * jitter, row[value_col], color=color, s=30, alpha=0.6, zorder=4)


def plot_mean_std(ax, agg, color, label, linewidth=2.0):
    means = agg["mean"].values
    stds = agg["std"].fillna(0).values
    ax.plot(x_pos, means, marker="o", markersize=6, linewidth=linewidth, color=color, label=label, zorder=5)
    ax.fill_between(x_pos, means - stds, means + stds, alpha=0.2, color=color)


def finish_ax(ax, title):
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FLIP_LABELS, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Flip rate", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, frameon=False)


def save_fig(fig, name):
    out = os.path.join(SAVE_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ------------------------------------------------------------------ #
# Figure 1a: scIB Bio conservation — NicheVI
# ------------------------------------------------------------------ #
for df, color, model_name, fname in [
    (nichevi_scib_df, NICHEVI_COLOR, "NicheVI", "scib_robustness_nichevi_f0f10.png"),
    (scanvi_scib_df, SCANVI_COLOR, "scANVI", "scib_robustness_scanvi_f0f10.png"),
]:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sub = df[["flip_name", "seed", "Bio conservation"]].dropna(subset=["Bio conservation"])
    agg = agg_by_flip(sub, "Bio conservation")
    scatter_seeds(ax, sub, "Bio conservation", color=color)
    plot_mean_std(ax, agg, color=color, label="mean ± std")
    finish_ax(ax, "Bio conservation")
    ax.set_ylabel("Score", fontsize=9)
    fig.suptitle(f"{model_name} — Bio conservation vs. flip rate\n(validation cells, f0–f10, 3 seeds)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, fname)

# ------------------------------------------------------------------ #
# Figure 2: NicheVI loss diagnostics (all 4 metrics)
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, len(NICHEVI_LOSS_METRICS), figsize=(5 * len(NICHEVI_LOSS_METRICS), 4.5))

for ax, (_, _, label) in zip(axes, NICHEVI_LOSS_METRICS):
    sub = nichevi_loss_df[nichevi_loss_df["metric"] == label]
    plot_mean_std(ax, agg_by_flip(sub, "min_val"), VAL_COLOR, "Val (min)")
    plot_mean_std(ax, agg_by_flip(sub, "train_at_best"), TRAIN_COLOR, "Train @ best val epoch")
    finish_ax(ax, label)
    ax.set_ylabel("Loss", fontsize=9)

fig.suptitle("NicheVI — Val (min) vs. Train @ best val epoch\n(f0–f10, 3 seeds)", fontsize=13)
fig.tight_layout()
save_fig(fig, "loss_robustness_f0f10.png")

# ------------------------------------------------------------------ #
# Figure 3: scANVI loss diagnostics (reconstruction + classification)
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, len(SCANVI_LOSS_METRICS), figsize=(5 * len(SCANVI_LOSS_METRICS), 4.5))

for ax, (_, _, label) in zip(axes, SCANVI_LOSS_METRICS):
    sub = scanvi_loss_df[scanvi_loss_df["metric"] == label]
    plot_mean_std(ax, agg_by_flip(sub, "min_val"), VAL_COLOR, "Val (min)")
    plot_mean_std(ax, agg_by_flip(sub, "train_at_best"), TRAIN_COLOR, "Train @ best val epoch")
    finish_ax(ax, label)
    ax.set_ylabel("Loss", fontsize=9)

fig.suptitle("scANVI — Val (min) vs. Train @ best val epoch\n(f0–f10, 3 seeds)", fontsize=13)
fig.tight_layout()
save_fig(fig, "loss_robustness_scanvi_f0f10.png")

# ------------------------------------------------------------------ #
# Figure 4: NicheVI overfitting gap
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, len(NICHEVI_LOSS_METRICS), figsize=(5 * len(NICHEVI_LOSS_METRICS), 4.5))

for ax, (_, _, label) in zip(axes, NICHEVI_LOSS_METRICS):
    sub = nichevi_loss_df[nichevi_loss_df["metric"] == label]
    plot_mean_std(ax, agg_by_flip(sub, "overfit_gap"), GAP_COLOR, "mean ± std")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    finish_ax(ax, label)
    ax.set_ylabel("Val − Train gap", fontsize=9)

fig.suptitle("NicheVI — Overfitting gap (val min − train @ best val epoch)\n(f0–f10, 3 seeds)", fontsize=13)
fig.tight_layout()
save_fig(fig, "overfit_gap_f0f10.png")

# ------------------------------------------------------------------ #
# Figure 5: scANVI overfitting gap
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, len(SCANVI_LOSS_METRICS), figsize=(5 * len(SCANVI_LOSS_METRICS), 4.5))

for ax, (_, _, label) in zip(axes, SCANVI_LOSS_METRICS):
    sub = scanvi_loss_df[scanvi_loss_df["metric"] == label]
    plot_mean_std(ax, agg_by_flip(sub, "overfit_gap"), GAP_COLOR, "mean ± std")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    finish_ax(ax, label)
    ax.set_ylabel("Val − Train gap", fontsize=9)

fig.suptitle("scANVI — Overfitting gap (val min − train @ best val epoch)\n(f0–f10, 3 seeds)", fontsize=13)
fig.tight_layout()
save_fig(fig, "overfit_gap_scanvi_f0f10.png")

print("\nAll done.")
