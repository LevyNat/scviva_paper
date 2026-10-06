"""
Aggregate scIB results across seeds and create plots with error bars.

This script:
1. Loads per-seed scIB results
2. Computes mean ± std across seeds for each flip rate
3. Creates line plots with error bars/shaded regions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from params import setup, SEEDS_TO_TEST, FLIP_RATES, ABLATION_CONFIGS

# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"

FIGURES_FOLDER = setup.FIGURES_FOLDER
# Main experiment results may be in the parent folder if FIGURES_FOLDER was changed for ablation
MAIN_FIGURES_FOLDER = "/home/nathanl/scviva_paper/crc_visium_hd/figures/robustness/seeds/"


def parse_embedding_name(name: str) -> dict:
    """Parse embedding name to extract flip rate, seed, and ablation type."""
    # Example: f0_s10_scanvi_lr0.0005_poisson_seed34_X_nicheVI_seed34_flip0.0
    # or: X_scanvi_seed34_flip0.0
    # or: eta0_f0_scanvi_lr0.0005_poisson_seed34_X_nicheVI_seed34_flip0.0

    result = {"flip_rate": None, "seed": None, "model": None, "flip_name": None, "ablation": None}

    # Extract flip rate
    flip_match = re.search(r"_flip([\d.]+)", name)
    if flip_match:
        result["flip_rate"] = float(flip_match.group(1))

    # Extract seed (take the last one if multiple)
    seed_matches = re.findall(r"seed(\d+)", name)
    if seed_matches:
        result["seed"] = int(seed_matches[-1])

    # Check for ablation prefix
    for ablation_name in ABLATION_CONFIGS:
        if name.startswith(f"{ablation_name}_"):
            result["ablation"] = ablation_name
            break

    # Determine model type
    if "nicheVI" in name or "nichevi" in name.lower():
        result["model"] = "NicheVI"
    elif "scanvi" in name.lower():
        result["model"] = "scANVI"

    # Extract flip name (f0, f1, etc.) - handle ablation prefix
    if result["ablation"]:
        flip_name_match = re.search(rf"^{result['ablation']}_(f\d+)_", name)
    else:
        flip_name_match = re.search(r"^(f\d+)_", name)

    if flip_name_match:
        result["flip_name"] = flip_name_match.group(1)
    else:
        # Infer from flip rate
        for fn, fr in FLIP_RATES.items():
            if result["flip_rate"] is not None and abs(fr - result["flip_rate"]) < 0.001:
                result["flip_name"] = fn
                break

    return result


def load_and_aggregate_results(result_type: str = "val", model: str = "nichevi"):
    """
    Load scIB results from all seeds and aggregate.

    Parameters
    ----------
    result_type : str
        "val" for validation results, "train" for training results
    model : str
        "nichevi" or "scanvi"

    Returns
    -------
    aggregated_df : pd.DataFrame
        DataFrame with mean and std for each flip rate
    """
    all_results = []

    for seed in SEEDS_TO_TEST:
        if model == "nichevi":
            filename = f"scib_{result_type}_results_robustness_seed{seed}.csv"
        else:
            filename = f"scib_{result_type}_results_robustness_scanvi_seed{seed}.csv"

        filepath = os.path.join(MAIN_FIGURES_FOLDER, filename)

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        df = pd.read_csv(filepath, index_col=0)

        # Parse each embedding and extract info
        for embedding in df.index:
            if embedding == "Metric Type":
                continue

            info = parse_embedding_name(embedding)

            if info["flip_name"] is None:
                continue

            row_data = {
                "seed": seed,
                "flip_name": info["flip_name"],
                "flip_rate": info["flip_rate"],
                "model": model,
            }

            # Add all metrics
            for col in df.columns:
                if col != "Metric Type":
                    try:
                        row_data[col] = float(df.loc[embedding, col])
                    except (ValueError, TypeError):
                        pass

            all_results.append(row_data)

    if not all_results:
        return None, None

    results_df = pd.DataFrame(all_results)

    # Aggregate by flip_name
    agg_df = results_df.groupby("flip_name").agg(
        flip_rate=("flip_rate", "first"),
        **{f"{col}_mean": (col, "mean") for col in results_df.columns
           if col not in ["seed", "flip_name", "flip_rate", "model"]},
        **{f"{col}_std": (col, "std") for col in results_df.columns
           if col not in ["seed", "flip_name", "flip_rate", "model"]},
        n_seeds=("seed", "count"),
    ).reset_index()

    # Sort by flip rate
    agg_df = agg_df.sort_values("flip_rate")

    return agg_df, results_df


def plot_robustness_curve(
    agg_df: pd.DataFrame,
    metric: str = "Bio conservation",
    model_name: str = "NicheVI",
    result_type: str = "Validation",
    save_path: str = None,
):
    """Plot robustness curve with error bars."""

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in agg_df.columns:
        print(f"Metric {metric} not found in results")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = agg_df["flip_rate"].values * 100  # Convert to percentage
    y = agg_df[mean_col].values
    yerr = agg_df[std_col].values

    # Plot with error bars
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)

    # Fill between for shaded region
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_xlabel("Label Flip Rate (%)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{model_name} - {result_type} Set\n{metric} vs Label Corruption", fontsize=14)

    # Add x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in x])

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.close()
    return fig


def plot_comparison(
    nichevi_agg: pd.DataFrame,
    scanvi_agg: pd.DataFrame,
    metric: str = "Bio conservation",
    result_type: str = "Validation",
    save_path: str = None,
):
    """Plot NicheVI vs scANVI comparison with error bars."""

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(10, 6))

    # NicheVI
    if nichevi_agg is not None and mean_col in nichevi_agg.columns:
        x = nichevi_agg["flip_rate"].values * 100
        y = nichevi_agg[mean_col].values
        yerr = nichevi_agg[std_col].values
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2,
                   linewidth=2, markersize=8, label="NicheVI", color="tab:blue")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="tab:blue")

    # scANVI
    if scanvi_agg is not None and mean_col in scanvi_agg.columns:
        x = scanvi_agg["flip_rate"].values * 100
        y = scanvi_agg[mean_col].values
        yerr = scanvi_agg[std_col].values
        ax.errorbar(x, y, yerr=yerr, fmt='s--', capsize=5, capthick=2,
                   linewidth=2, markersize=8, label="scANVI", color="tab:orange")
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="tab:orange")

    ax.set_xlabel("Label Flip Rate (%)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{result_type} Set - {metric} vs Label Corruption", fontsize=14)

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Set x-ticks
    if nichevi_agg is not None:
        x_ticks = nichevi_agg["flip_rate"].values * 100
    else:
        x_ticks = scanvi_agg["flip_rate"].values * 100
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(v)}%" for v in x_ticks])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")

    plt.close()
    return fig


def load_and_aggregate_fixed_sample(model: str = "nichevi"):
    """
    Load fixed sample scIB results and aggregate by flip rate.

    The fixed sample uses the same cells for all seeds, so this is the
    cleanest comparison - only training variance (label flipping) differs.

    Parameters
    ----------
    model : str
        "nichevi" or "scanvi"

    Returns
    -------
    aggregated_df : pd.DataFrame
        DataFrame with mean and std for each flip rate
    """
    if model == "nichevi":
        filename = "scib_fixed_sample_results_nichevi.csv"
    else:
        filename = "scib_fixed_sample_results_scanvi.csv"

    filepath = os.path.join(MAIN_FIGURES_FOLDER, filename)

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None, None

    df = pd.read_csv(filepath, index_col=0)

    all_results = []

    # Parse each embedding and extract info
    for embedding in df.index:
        if embedding == "Metric Type":
            continue

        info = parse_embedding_name(embedding)

        if info["flip_name"] is None:
            continue

        row_data = {
            "seed": info["seed"],
            "flip_name": info["flip_name"],
            "flip_rate": info["flip_rate"],
            "model": model,
        }

        # Add all metrics
        for col in df.columns:
            if col != "Metric Type":
                try:
                    row_data[col] = float(df.loc[embedding, col])
                except (ValueError, TypeError):
                    pass

        all_results.append(row_data)

    if not all_results:
        return None, None

    results_df = pd.DataFrame(all_results)

    # Aggregate by flip_name (across seeds)
    agg_df = results_df.groupby("flip_name").agg(
        flip_rate=("flip_rate", "first"),
        **{f"{col}_mean": (col, "mean") for col in results_df.columns
           if col not in ["seed", "flip_name", "flip_rate", "model"]},
        **{f"{col}_std": (col, "std") for col in results_df.columns
           if col not in ["seed", "flip_name", "flip_rate", "model"]},
        n_seeds=("seed", "count"),
    ).reset_index()

    # Sort by flip rate
    agg_df = agg_df.sort_values("flip_rate")

    return agg_df, results_df


def load_and_aggregate_ablation():
    """
    Load ablation fixed sample results and aggregate by ablation type and flip rate.

    Returns
    -------
    dict mapping ablation_name -> (agg_df, raw_df)
    """
    filename = "scib_fixed_sample_results_ablation.csv"
    filepath = os.path.join(FIGURES_FOLDER, filename)

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return {}

    df = pd.read_csv(filepath, index_col=0)

    # Collect results per ablation type
    results_by_ablation = {name: [] for name in ABLATION_CONFIGS}

    for embedding in df.index:
        if embedding == "Metric Type":
            continue

        info = parse_embedding_name(embedding)
        if info["flip_name"] is None or info["ablation"] is None:
            continue

        row_data = {
            "seed": info["seed"],
            "flip_name": info["flip_name"],
            "flip_rate": info["flip_rate"],
            "ablation": info["ablation"],
        }

        for col in df.columns:
            if col != "Metric Type":
                try:
                    row_data[col] = float(df.loc[embedding, col])
                except (ValueError, TypeError):
                    pass

        results_by_ablation[info["ablation"]].append(row_data)

    output = {}
    for ablation_name, results in results_by_ablation.items():
        if not results:
            continue

        results_df = pd.DataFrame(results)
        agg_df = results_df.groupby("flip_name").agg(
            flip_rate=("flip_rate", "first"),
            **{f"{col}_mean": (col, "mean") for col in results_df.columns
               if col not in ["seed", "flip_name", "flip_rate", "ablation"]},
            **{f"{col}_std": (col, "std") for col in results_df.columns
               if col not in ["seed", "flip_name", "flip_rate", "ablation"]},
            n_seeds=("seed", "count"),
        ).reset_index()
        agg_df = agg_df.sort_values("flip_rate")
        output[ablation_name] = (agg_df, results_df)

    return output


def plot_ablation_comparison(
    nichevi_agg: pd.DataFrame,
    scanvi_agg: pd.DataFrame,
    ablation_aggs: dict,
    metric: str = "Bio conservation",
    result_type: str = "Fixed Sample",
    save_path: str = None,
):
    """Plot NicheVI vs scANVI vs ablations comparison with error bars."""

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Style config for each model
    styles = {
        "NicheVI": {"fmt": "o-", "color": "tab:blue"},
        "scANVI": {"fmt": "s--", "color": "tab:orange"},
        "eta0": {"fmt": "^-.", "color": "tab:green"},
        "s0": {"fmt": "D:", "color": "tab:red"},
    }

    labels = {
        "NicheVI": "NicheVI (full)",
        "scANVI": "scANVI",
        "eta0": r"NicheVI ($\eta$=0)",
        "s0": "NicheVI (no spatial)",
    }

    # NicheVI
    if nichevi_agg is not None and mean_col in nichevi_agg.columns:
        x = nichevi_agg["flip_rate"].values * 100
        y = nichevi_agg[mean_col].values
        yerr = nichevi_agg[std_col].values
        s = styles["NicheVI"]
        ax.errorbar(x, y, yerr=yerr, fmt=s["fmt"], capsize=4, capthick=1.5,
                     linewidth=2, markersize=7, label=labels["NicheVI"], color=s["color"])
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=s["color"])

    # scANVI
    if scanvi_agg is not None and mean_col in scanvi_agg.columns:
        x = scanvi_agg["flip_rate"].values * 100
        y = scanvi_agg[mean_col].values
        yerr = scanvi_agg[std_col].values
        s = styles["scANVI"]
        ax.errorbar(x, y, yerr=yerr, fmt=s["fmt"], capsize=4, capthick=1.5,
                     linewidth=2, markersize=7, label=labels["scANVI"], color=s["color"])
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=s["color"])

    # Ablations
    for ablation_name, (abl_agg, _) in ablation_aggs.items():
        if abl_agg is not None and mean_col in abl_agg.columns:
            x = abl_agg["flip_rate"].values * 100
            y = abl_agg[mean_col].values
            yerr = abl_agg[std_col].values
            s = styles.get(ablation_name, {"fmt": "x-", "color": "tab:purple"})
            ax.errorbar(x, y, yerr=yerr, fmt=s["fmt"], capsize=4, capthick=1.5,
                         linewidth=2, markersize=7, label=labels.get(ablation_name, ablation_name),
                         color=s["color"])
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=s["color"])

    ax.set_xlabel("Label Flip Rate (%)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{result_type} - {metric} vs Label Corruption", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    if nichevi_agg is not None:
        x_ticks = nichevi_agg["flip_rate"].values * 100
    elif scanvi_agg is not None:
        x_ticks = scanvi_agg["flip_rate"].values * 100
    else:
        x_ticks = list(ablation_aggs.values())[0][0]["flip_rate"].values * 100
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(v)}%" for v in x_ticks])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
        print(f"Saved ablation comparison plot to {save_path}")

    plt.close()
    return fig


def main():
    """Main aggregation and plotting."""

    print("=" * 60)
    print("Aggregating scIB results across seeds")
    print("=" * 60)

    # ===== Fixed Sample Results (cleanest comparison) =====
    print("\n--- Fixed Sample Results (Same Cells, Different Training Seeds) ---")

    nichevi_fixed_agg, nichevi_fixed_raw = load_and_aggregate_fixed_sample("nichevi")
    scanvi_fixed_agg, scanvi_fixed_raw = load_and_aggregate_fixed_sample("scanvi")

    if nichevi_fixed_agg is not None:
        print("\nNicheVI Fixed Sample (aggregated):")
        print(nichevi_fixed_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        nichevi_fixed_agg.to_csv(os.path.join(FIGURES_FOLDER, "nichevi_fixed_sample_aggregated.csv"), index=False)

    if scanvi_fixed_agg is not None:
        print("\nscANVI Fixed Sample (aggregated):")
        print(scanvi_fixed_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        scanvi_fixed_agg.to_csv(os.path.join(FIGURES_FOLDER, "scanvi_fixed_sample_aggregated.csv"), index=False)

    # Load and aggregate validation results
    print("\n--- Validation Results (Per-Seed Splits) ---")

    nichevi_val_agg, nichevi_val_raw = load_and_aggregate_results("val", "nichevi")
    scanvi_val_agg, scanvi_val_raw = load_and_aggregate_results("val", "scanvi")

    if nichevi_val_agg is not None:
        print("\nNicheVI Validation (aggregated):")
        print(nichevi_val_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        nichevi_val_agg.to_csv(os.path.join(FIGURES_FOLDER, "nichevi_val_aggregated.csv"), index=False)

    if scanvi_val_agg is not None:
        print("\nscANVI Validation (aggregated):")
        print(scanvi_val_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        scanvi_val_agg.to_csv(os.path.join(FIGURES_FOLDER, "scanvi_val_aggregated.csv"), index=False)

    # Load and aggregate training results
    print("\n--- Training Results ---")

    nichevi_train_agg, nichevi_train_raw = load_and_aggregate_results("train", "nichevi")
    scanvi_train_agg, scanvi_train_raw = load_and_aggregate_results("train", "scanvi")

    if nichevi_train_agg is not None:
        print("\nNicheVI Training (aggregated):")
        print(nichevi_train_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        nichevi_train_agg.to_csv(os.path.join(FIGURES_FOLDER, "nichevi_train_aggregated.csv"), index=False)

    if scanvi_train_agg is not None:
        print("\nscANVI Training (aggregated):")
        print(scanvi_train_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        scanvi_train_agg.to_csv(os.path.join(FIGURES_FOLDER, "scanvi_train_aggregated.csv"), index=False)

    # ===== Ablation Results (fixed sample only) =====
    print("\n--- Ablation Results (Fixed Sample) ---")

    ablation_aggs = load_and_aggregate_ablation()

    for ablation_name, (abl_agg, abl_raw) in ablation_aggs.items():
        print(f"\n{ablation_name} (aggregated):")
        print(abl_agg[["flip_name", "flip_rate", "Bio conservation_mean", "Bio conservation_std", "n_seeds"]])
        abl_agg.to_csv(os.path.join(FIGURES_FOLDER, f"ablation_{ablation_name}_fixed_sample_aggregated.csv"), index=False)

    # Generate plots
    print("\n--- Generating Plots ---")

    metrics = ["Bio conservation", "Leiden NMI", "Leiden ARI", "Silhouette label", "cLISI"]

    for metric in metrics:
        # Validation comparison
        plot_comparison(
            nichevi_val_agg, scanvi_val_agg,
            metric=metric,
            result_type="Validation",
            save_path=os.path.join(FIGURES_FOLDER, f"comparison_val_{metric.replace(' ', '_').lower()}.png")
        )

        # Training comparison
        plot_comparison(
            nichevi_train_agg, scanvi_train_agg,
            metric=metric,
            result_type="Training",
            save_path=os.path.join(FIGURES_FOLDER, f"comparison_train_{metric.replace(' ', '_').lower()}.png")
        )

        # Fixed sample comparison (same cells, different training seeds)
        plot_comparison(
            nichevi_fixed_agg, scanvi_fixed_agg,
            metric=metric,
            result_type="Fixed Sample",
            save_path=os.path.join(FIGURES_FOLDER, f"comparison_fixed_{metric.replace(' ', '_').lower()}.png")
        )

        # Ablation comparison (NicheVI full vs eta0 vs s0 vs scANVI)
        if ablation_aggs:
            plot_ablation_comparison(
                nichevi_fixed_agg, scanvi_fixed_agg, ablation_aggs,
                metric=metric,
                result_type="Ablation (Fixed Sample)",
                save_path=os.path.join(FIGURES_FOLDER, f"comparison_ablation_{metric.replace(' ', '_').lower()}.png")
            )

    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
