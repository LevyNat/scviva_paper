"""Label-free choice of the Leiden resolution for the brain unlabeled run (Fig. S2).

Applies the rule from the tonsil methods (slidetags_tonsil/tonsil/runner_no_labels.py,
NL_RES_SELECT=internal) to the brain scVI embedding, with no cell-type annotations involved:
  * Leiden on a 15-NN graph of the scVI latent (igraph, n_iterations=2, undirected), exactly as in
    runner_no_labels.py, for resolutions 0.05 .. 2.00 in steps of 0.05;
  * cluster-quality scores on the scVI latent: silhouette (max), Calinski-Harabasz (max),
    Davies-Bouldin (min).
Two selection rules are reported:
  * "silhouette": the resolution maximising silhouette (the rule as written in the paper's
    tonsil methods);
  * "vote": majority vote of the three scores, silhouette breaking ties (the rule as coded in the
    tonsil runner).
By default (SIL_SAMPLE=0) the silhouette is exact: computed on all cells, in chunks on the GPU
(silhouette_exact below; checked against scikit-learn on a subsample at start-up). With SIL_SAMPLE > 0 it
is instead the mean of scikit-learn's silhouette on seeded subsamples of that size (seeds SIL_SEEDS).
CH and DB use all cells.

Sanity check: the cluster counts at the resolutions used for the paper (0.1, 0.15, 0.2, 0.3) are
compared with the number of label categories stored in the four trained unlabeled scVIVA models.

Reads only; writes one CSV into FIGURES_FOLDER. Run in the legacy env:
    cd merfish_brain/unlabeled && python select_resolution.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from params import setup

scvi.settings.seed = 34

# the trained models (read-only); override with env var to point at another copy
CKPT = os.environ.get(
    "SCVI_CKPT_DIR",
    "/home/nathanl/scviva_paper/merfish_brain/checkpoints/adata_M1_M2_core_6_sections",
)
SIL_SAMPLE = int(os.environ.get("SIL_SAMPLE", 0))  # 0 = exact silhouette on all cells
SIL_SEEDS = [0, 1, 2]
PAPER_RES = [0.1, 0.15, 0.2, 0.3]
OUT = os.path.join(setup.FIGURES_FOLDER, "leiden_resolution_selection_brain.csv")

adata = ad.read_h5ad(os.path.join(setup.DATA_FOLDER, setup.DATA_FILE))
adata.layers["counts"] = adata.raw.X.copy()

# same scVI model and embedding as runner_no_labels.py (it sets EXPRESSION_MODEL = "scvi")
scvi_path = os.path.join(CKPT, f"scvivae_E{setup.N_EPOCHS_SCVI}_{setup.LIKELIHOOD}.pt")
model = scvi.model.SCVI.load(dir_path=scvi_path, adata=adata)
X = model.get_latent_representation(batch_size=setup.BATCH_SIZE_SCVI)
adata.obsm["X_scVI"] = X
print("scVI latent:", X.shape)

sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15, key_added="scvi_neighbors")


def silhouette_exact(X, labels, chunk=2048):
    """Mean silhouette over all cells (Euclidean), as sklearn.metrics.silhouette_score without sampling."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    _, lab = np.unique(labels, return_inverse=True)
    lab = torch.as_tensor(lab, device=device)
    k = int(lab.max()) + 1
    counts = torch.bincount(lab, minlength=k).double()
    s = torch.empty(len(lab), dtype=torch.float64, device=device)
    for start in range(0, len(lab), chunk):
        stop = min(start + chunk, len(lab))
        d = torch.cdist(Xt[start:stop], Xt).double()  # (chunk, n)
        sums = torch.zeros(stop - start, k, dtype=torch.float64, device=device).index_add_(1, lab, d)
        own = lab[start:stop]
        own_n = counts[own]
        a = sums.gather(1, own[:, None]).squeeze(1) / (own_n - 1).clamp(min=1)
        mean_other = sums / counts
        mean_other.scatter_(1, own[:, None], float("inf"))
        b = mean_other.min(dim=1).values
        s_chunk = (b - a) / torch.maximum(a, b)
        s[start:stop] = torch.where(own_n > 1, s_chunk, torch.zeros_like(s_chunk))  # singletons score 0
    return float(s.mean())


# check silhouette_exact against scikit-learn on a subsample
_idx = np.random.default_rng(0).choice(len(X), 5000, replace=False)
_lab = np.random.default_rng(1).integers(0, 12, 5000)
_ref, _ours = silhouette_score(X[_idx], _lab), silhouette_exact(X[_idx], _lab)
print(f"silhouette_exact check vs scikit-learn: {_ours:.8f} vs {_ref:.8f}")
assert abs(_ours - _ref) < 1e-5


def leiden(res):
    sc.tl.leiden(adata, resolution=res, neighbors_key="scvi_neighbors", key_added="_probe",
                 flavor="igraph", n_iterations=2, directed=False)
    return adata.obs["_probe"].values


rows = []
for step in range(1, 41):  # 0.05 .. 2.00
    res = round(0.05 * step, 2)
    lab = leiden(res)
    k = len(np.unique(lab))
    row = {"resolution": res, "n_clusters": k}
    if k >= 2:
        if SIL_SAMPLE > 0:
            sils = [silhouette_score(X, lab, sample_size=min(SIL_SAMPLE, len(lab)), random_state=s) for s in SIL_SEEDS]
            row.update({f"silhouette_seed{s}": v for s, v in zip(SIL_SEEDS, sils)})
            row["silhouette"] = float(np.mean(sils))
        else:
            row["silhouette"] = silhouette_exact(X, lab)
        row["calinski_harabasz"] = calinski_harabasz_score(X, lab)
        row["davies_bouldin"] = davies_bouldin_score(X, lab)
    rows.append(row)
    print(f"res={res:.2f} k={k:3d} sil={row.get('silhouette', float('nan')):.4f} "
          f"CH={row.get('calinski_harabasz', float('nan')):.0f} DB={row.get('davies_bouldin', float('nan')):.3f}")

sw = pd.DataFrame(rows)
valid = sw.dropna(subset=["silhouette"])
sil_pick = float(valid.loc[valid.silhouette.idxmax(), "resolution"])
votes = [sil_pick,
         float(valid.loc[valid.calinski_harabasz.idxmax(), "resolution"]),
         float(valid.loc[valid.davies_bouldin.idxmin(), "resolution"])]
vc = pd.Series(votes).value_counts()
vote_pick = float(vc.index[0]) if vc.iloc[0] >= 2 else votes[0]
per_seed_picks = (
    {s: float(valid.loc[valid[f"silhouette_seed{s}"].idxmax(), "resolution"]) for s in SIL_SEEDS} if SIL_SAMPLE > 0 else "exact"
)

# cluster counts the four trained unlabeled models were registered with
reg_counts = {}
for res in PAPER_RES:
    p = os.path.join(CKPT, f"nichevae_{res}_s10_scvi_lr{setup.LR_NICHEVI}_{setup.LIKELIHOOD}_{setup.N_EPOCHS_NICHEVI}.pt", "model.pt")
    reg = torch.load(p, map_location="cpu", weights_only=False)["attr_dict"]["registry_"]
    reg_counts[res] = len(reg["field_registries"]["labels"]["state_registry"]["categorical_mapping"])

sw["rule_silhouette_pick"] = sw.resolution == sil_pick
sw["rule_vote_pick"] = sw.resolution == vote_pick
sw["n_clusters_in_trained_model"] = sw.resolution.map(reg_counts)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)
sw.to_csv(OUT, index=False)

print("\n=== summary ===")
print(f"silhouette rule (paper text): res {sil_pick:.2f} "
      f"({int(sw.set_index('resolution').loc[sil_pick, 'n_clusters'])} clusters); per-seed picks {per_seed_picks}")
print(f"vote rule (tonsil code): votes (sil, CH, DB) = {votes} -> res {vote_pick:.2f} "
      f"({int(sw.set_index('resolution').loc[vote_pick, 'n_clusters'])} clusters)")
for res in PAPER_RES:
    k_now = int(sw.set_index("resolution").loc[res, "n_clusters"])
    print(f"res {res}: {k_now} clusters now, {reg_counts[res]} in trained model -> {'OK' if k_now == reg_counts[res] else 'MISMATCH'}")
print("wrote", OUT)
