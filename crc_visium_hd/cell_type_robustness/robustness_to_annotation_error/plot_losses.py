"""
Plot training loss curves from saved checkpoints.

Loads history directly from model.pt files (no full model loading needed).
Plots train/val on same figure, one figure per metric, subplots per flip rate.
Separate figures for NicheVI (full), eta0, and s0.

Usage:
    python plot_losses.py              # Plot seed 34 (default)
    python plot_losses.py --seed 42    # Plot a different seed
"""

import argparse
import os
import matplotlib.pyplot as plt
import torch

from params import setup, FLIP_RATES, ABLATION_CONFIGS, SEEDS_TO_TEST

plt.rcParams["svg.fonttype"] = "none"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=SEEDS_TO_TEST[0], help="Seed to plot")
args = parser.parse_args()

SEED = args.seed
CHECKPOINT_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/adata_legacy_hvg4k_robustness"
SAVE_DIR = "/home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/losses/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Metrics to plot: (display_name, train_key, val_key)
METRICS = [
    ("ELBO", "elbo_train", "elbo_validation"),
    ("Niche Composition", "niche_compo_train", "niche_compo_validation"),
    ("Niche Reconstruction", "niche_reconst_train", "niche_reconst_validation"),
    ("KL Local", "kl_local_train", "kl_local_validation"),
    ("Reconstruction Loss", "reconstruction_loss_train", "reconstruction_loss_validation"),
]

# Model groups: name -> checkpoint filename pattern
MODEL_GROUPS = {
    "NicheVI": {
        flip_name: f"nichevae_{flip_name}_s10_{setup.EXPRESSION_MODEL}_lr{setup.LR_NICHEVI}_{setup.LIKELIHOOD}_seed{SEED}_E{setup.N_EPOCHS_NICHEVI}_seed{SEED}_flip{flip_rate}.pt"
        for flip_name, flip_rate in FLIP_RATES.items()
    },
}

for ablation_name in ABLATION_CONFIGS:
    MODEL_GROUPS[ablation_name] = {
        flip_name: f"nichevae_{ablation_name}_{flip_name}_{setup.EXPRESSION_MODEL}_lr{setup.LR_NICHEVI}_{setup.LIKELIHOOD}_seed{SEED}_E{setup.N_EPOCHS_NICHEVI}_seed{SEED}_flip{flip_rate}.pt"
        for flip_name, flip_rate in FLIP_RATES.items()
    }


def load_history(checkpoint_name):
    """Load history dict from a checkpoint without loading the full model."""
    path = os.path.join(CHECKPOINT_DIR, checkpoint_name, "model.pt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["attr_dict"]["history_"]


def plot_group(group_name, flip_checkpoints, metric_name, train_key, val_key):
    """Plot a single metric for a model group across all flip rates."""
    flip_names = list(FLIP_RATES.keys())
    n = len(flip_names)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for idx, flip_name in enumerate(flip_names):
        ax = axes[idx // n_cols][idx % n_cols]
        ckpt_name = flip_checkpoints.get(flip_name)
        if ckpt_name is None:
            ax.set_visible(False)
            continue

        hist = load_history(ckpt_name)
        if hist is None:
            ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(flip_name, fontsize=10)
            continue

        if train_key in hist:
            df = hist[train_key]
            ax.plot(df.index, df.iloc[:, 0].values, label="train", color="darkgreen", linewidth=1.2)
        if val_key in hist:
            df = hist[val_key]
            ax.plot(df.index, df.iloc[:, 0].values, label="val", color="firebrick", linewidth=1.2)

        ax.set_title(flip_name, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Epoch")

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    label = group_name if group_name == "NicheVI" else f"NicheVI ({group_name})"
    fig.suptitle(f"{label} — {metric_name} (seed {SEED})", fontsize=13)
    fig.tight_layout()

    save_name = f"{group_name}_{metric_name.replace(' ', '_').lower()}_seed{SEED}"
    fig.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(SAVE_DIR, f"{save_name}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_name}")


def main():
    print(f"Plotting loss curves for seed {SEED}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Save dir: {SAVE_DIR}")
    print()

    for group_name, flip_checkpoints in MODEL_GROUPS.items():
        print(f"--- {group_name} ---")
        for metric_name, train_key, val_key in METRICS:
            plot_group(group_name, flip_checkpoints, metric_name, train_key, val_key)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
