"""Utilities for creating synthetic niche-correlated genes."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

from params import setup


def _build_synthetic_matrix(
    alpha: np.ndarray,
    *,
    n_genes: int,
    signal_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic genes correlated with alpha."""
    lower, upper = signal_bounds
    signal_strengths = np.linspace(lower, upper, n_genes)
    synthetic_genes = []
    correlations = []

    for strength in signal_strengths:
        noise = rng.normal(0, 1, size=alpha.shape[0])
        gene = np.clip(np.round(strength * alpha + noise), a_min=0, a_max=None)
        synthetic_genes.append(gene)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                r, _ = pearsonr(alpha, gene)
            except Exception:
                r = np.nan
        correlations.append(float(r) if np.isfinite(r) else float("nan"))

    gene_matrix = np.column_stack(synthetic_genes)
    return gene_matrix, signal_strengths, np.asarray(correlations, dtype=float)


def add_synthetic_correlated_genes(
    adata: ad.AnnData,
    target_cell_type: str,
    *,
    n_genes: int = 10,
    signal_bounds: Tuple[float, float] = (1.0, 2.0),
    random_seed: int | None = None,
    composition_key: str = "neighborhood_composition",
    var_prefix: str = "s",
    copy: bool = True,
    update_raw: bool = False,
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    """Return a new AnnData augmented with niche-correlated synthetic genes."""
    if not copy:
        warnings.warn(
            "copy=False is not yet supported; returning a new AnnData with the synthetic genes instead."
        )

    if n_genes <= 0:
        raise ValueError("n_genes must be a positive integer.")

    if composition_key not in adata.obsm:
        raise KeyError(f"'{composition_key}' not found in adata.obsm.")

    if setup.CELL_TYPE not in adata.obs:
        raise KeyError(f"'{setup.CELL_TYPE}' not found in adata.obs.")

    cell_types = adata.obs[setup.CELL_TYPE].astype(str)
    mask = cell_types == target_cell_type
    if mask.sum() == 0:
        raise ValueError(f"No cells annotated as '{target_cell_type}' were found.")

    comp_df = adata.obsm[composition_key]
    mean_composition = comp_df.loc[mask].mean(axis=0).fillna(0.0)
    dominant_cell_type = mean_composition.idxmax()
    alpha = comp_df[dominant_cell_type].to_numpy()

    rng = np.random.default_rng(random_seed)
    lower, upper = signal_bounds
    if lower > upper:
        raise ValueError("signal_bounds lower bound must not exceed the upper bound.")

    gene_matrix, signal_strengths, correlations = _build_synthetic_matrix(
        alpha,
        n_genes=n_genes,
        signal_bounds=signal_bounds,
        rng=rng,
    )

    target_dtype = getattr(adata.X, "dtype", None) or gene_matrix.dtype
    gene_matrix = gene_matrix.astype(target_dtype, copy=False)

    avg_expression = gene_matrix[mask].mean(axis=0)

    adata.var_names = adata.var["gene_name"]

    existing_names = {str(name) for name in adata.var_names}
    new_names = []
    for idx in range(n_genes):
        base = f"{var_prefix}{idx + 1}"
        candidate = base
        suffix = 1
        while candidate in existing_names or candidate in new_names:
            suffix += 1
            candidate = f"{base}_{suffix}"
        new_names.append(candidate)

    synthetic_var = pd.DataFrame(index=new_names)
    synthetic_var["is_synthetic"] = True

    synthetic_adata = ad.AnnData(
        X=gene_matrix,
        obs=adata.obs.copy(),
        var=synthetic_var,
        dtype=gene_matrix.dtype,
    )

    for layer_key, layer_value in adata.layers.items():
        if layer_value is None:
            continue
        layer_dtype = getattr(layer_value, "dtype", gene_matrix.dtype)
        if sparse.issparse(layer_value):
            synthetic_layer = sparse.csr_matrix(gene_matrix.astype(layer_dtype, copy=False))
        else:
            synthetic_layer = gene_matrix.astype(layer_dtype, copy=False)
        synthetic_adata.layers[layer_key] = synthetic_layer

    augmented = ad.concat(
        [adata, synthetic_adata],
        axis=1,
        join="outer",
        merge="same",
        label=None,
        index_unique=None,
    )

    if "is_synthetic" not in augmented.var.columns:
        augmented.var["is_synthetic"] = False
    augmented.var["is_synthetic"] = augmented.var["is_synthetic"].fillna(False)
    augmented.var.loc[new_names, "is_synthetic"] = True

    augmented.uns = deepcopy(adata.uns)
    info = {
        "dominant_cell_type": dominant_cell_type,
        "target_cell_type": target_cell_type,
        "var_names": new_names,
        "signal_strengths": signal_strengths.tolist(),
        "correlations": correlations.tolist(),
        "avg_expression": avg_expression.tolist(),
    }
    augmented.uns.setdefault("synthetic_genes", {})[target_cell_type] = info

    if adata.raw is not None and not update_raw:
        augmented.raw = adata.raw.to_adata()
    if update_raw:
        augmented.raw = augmented.copy()

    return augmented, info


def create_single_synthetic_alpha(
    adata: ad.AnnData,
    *,
    name: str = "alpha_synth",
    low: float = 0.0,
    high: float = 0.2,
    composition_key: str = "alpha",
    random_seed: int = 0,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Insert a new alpha component sampled uniformly in [low, high].
    Ensures synthetic alpha never becomes the major niche component.
    """
    rng = np.random.default_rng(random_seed)

    n = adata.n_obs
    alpha_synth = rng.uniform(low, high, size=n)

    # Retrieve existing alpha
    comp = pd.DataFrame(adata.obsm[composition_key]).copy()

    # Add the synthetic alpha
    comp[name] = alpha_synth

    if renormalize:
        comp = comp.div(comp.sum(axis=1), axis=0)

    adata.obsm[composition_key] = comp

    return alpha_synth

def add_single_synthetic_gene(
    adata: ad.AnnData,
    alpha: np.ndarray,
    *,
    strength: float = 2.0,
    noise_sd: float = 1.0,
    name: str = "synth_gene",
    random_seed: int = 0,
    update_raw: bool = False,
) -> tuple[ad.AnnData, float]:
    """
    Add a single synthetic gene correlated with the provided α vector.
    Returns (augmented_adata, pearson_correlation).
    """
    rng = np.random.default_rng(random_seed)
    n = adata.n_obs

    # Build expression: strength * alpha + noise
    noise = rng.normal(0, noise_sd, n)
    gene = np.clip(np.round(strength * alpha + noise), 0, None)

    # Compute correlation
    from scipy.stats import pearsonr

    r, _ = pearsonr(alpha, gene)

    # Make a small AnnData holding only this gene
    synth_var = pd.DataFrame(index=[name])
    synth_var["is_synthetic"] = True

    synth = ad.AnnData(
        X=gene.reshape(-1, 1),
        obs=adata.obs.copy(),
        var=synth_var,
    )

    # Concatenate with original
    aug = ad.concat([adata, synth], axis=1)

    if adata.raw is not None and not update_raw:
        aug.raw = adata.raw.to_adata()

    if update_raw:
        aug.raw = aug.copy()

    return aug, float(r)

_all__ = ["add_synthetic_correlated_genes"]
