"""Score one tonsil gene-panel benchmark: cell-type preservation + region conservation.

    python score_panel.py fullgenes     # or 15000hvg

For panel P this needs, all built from tonsil_nl_<P>.h5ad:
  * `./run_benchmark.sh slidetags_tonsil_<P> <gpu>`  -> BANKSY/SimVI embeddings in the data file
  * `python runner.py slidetags_tonsil/tonsil_<P>`         -> supervised scANVI + scVIVA
  * runner_no_labels.py with NL_DATA_FILE=tonsil_nl_<P>.h5ad -> scVI + unlabeled scVIVA
Same metrics as runner.py:
scib BioConservation vs cell_type, and region conservation = per-cell-type BioConservation of
cell_type_niche. Harmony is left out on purpose (single donor -> it is a no-op copy of BANKSY).
"""
import os
import sys
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation

warnings.filterwarnings("ignore")
plt.rcParams["svg.fonttype"] = "none"

ROOT = "/home/nathanl/scviva_paper/slidetags_tonsil"
PANEL = sys.argv[1] if len(sys.argv) > 1 else "fullgenes"
STEM = f"tonsil_nl_{PANEL}"
# optional variant: NL_H5AD = another unlabeled-run output, OUT_SUBDIR = where its results go
OUT = os.path.join(f"{ROOT}/figures/tonsil_{PANEL}", os.environ.get("OUT_SUBDIR", "")) + "/"
os.makedirs(OUT, exist_ok=True)

adata = ad.read_h5ad(f"/home/nathanl/Data/slidetags/{STEM}.h5ad")
nl = ad.read_h5ad(os.environ.get("NL_H5AD", f"{ROOT}/checkpoints/tonsil_nolabels/{STEM}/{STEM}_nicheVI.h5ad"))
ref = ad.read_h5ad(f"{ROOT}/checkpoints/tonsil_{PANEL}/{STEM}/{STEM}_nicheVI.h5ad")
# the unlabeled run trains exactly one nicheVI (NL_TARGET_CLUSTERS mode); find its key
nl_keys = [k for k in nl.obsm if k.endswith("_s10_scvi_lr0.0005_poisson_X_nicheVI")]
assert len(nl_keys) == 1, nl_keys
print("unlabeled scVIVA key:", nl_keys[0])
assert nl.obs_names.equals(adata.obs_names) and ref.obs_names.equals(adata.obs_names)

EMB = {  # obsm key in the scored object -> (source, key in source)
    "PCA": (adata, "banksy_pc_20_nonspatial"),
    "BANKSY (lambda 0.2)": (adata, "banksy_pc_20_lambda_0.2"),
    "SimVI": (adata, "simvi25_both_mae_50"),
    "scVI": (nl, "X_scVI"),
    "scVIVA (unlabeled)": (nl, nl_keys[0]),
    "scANVI": (ref, "X_scanvi"),
    "scVIVA": (ref, "s10_scanvi_lr0.0005_poisson_X_nicheVI"),
}
S = ad.AnnData(obs=adata.obs[["cell_type", "region", "donor_id"]].copy())
for name, (src, key) in EMB.items():
    S.obsm[name] = src.obsm[key]
keys = list(EMB)

biocons = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=True,
                          nmi_ari_cluster_labels_kmeans=False, silhouette_label=True, clisi_knn=True)

bm = Benchmarker(S, batch_key="donor_id", label_key="cell_type", embedding_obsm_keys=keys,
                 bio_conservation_metrics=biocons, batch_correction_metrics=None, n_jobs=-1,
                 pre_integrated_embedding_obsm_key="PCA")  # S has no genes: skip scib's own PCA
bm.benchmark()
ct = bm.get_results(min_max_scale=False, clean_names=True)
ct.to_csv(f"{OUT}scib_cell_type_results_{PANEL}.csv")

S.obs["cell_type_niche"] = S.obs["cell_type"].astype(str) + "_" + S.obs["region"].astype(str)
vc = S.obs["cell_type_niche"].value_counts()
sub = S[S.obs["cell_type_niche"].isin(vc[vc >= 30].index)].copy()  # TRESHOLD from tonsil/params.py
rows = {}
for c in sub.obs["cell_type"].unique().tolist():
    s = sub[sub.obs["cell_type"] == c].copy()
    if s.obs["cell_type_niche"].nunique() < 2:
        continue
    b = Benchmarker(s, batch_key="donor_id", label_key="cell_type_niche", embedding_obsm_keys=keys,
                    bio_conservation_metrics=biocons, batch_correction_metrics=None, n_jobs=-1,
                 pre_integrated_embedding_obsm_key="PCA")  # S has no genes: skip scib's own PCA
    b.benchmark()
    rows[c] = b.get_results(min_max_scale=False)["Bio conservation"]
reg = pd.DataFrame(rows).T.drop(columns="Metric Type", errors="ignore").astype(float)
reg.index.name = "cell_type"
reg.to_csv(f"{OUT}BioCons_cell_type_{PANEL}.csv")

summary = pd.DataFrame({
    "region_cons": reg.mean(), "region_std": reg.std(),
    "celltype_pres": ct.drop("Metric Type", errors="ignore")["Bio conservation"].astype(float),
}).sort_values("celltype_pres", ascending=False)
summary.to_csv(f"{OUT}region_celltype_summary_{PANEL}.csv")
print(summary.round(3).to_string())

colors = dict(zip(keys, ["#999999", "#377eb8", "#ff7f00", "#4daf4a", "#fb9a99", "#984ea3", "#e41a1c"]))
fig, ax = plt.subplots(figsize=(6.5, 5))
for k, r in summary.iterrows():
    ax.errorbar(r.region_cons, r.celltype_pres, xerr=r.region_std, fmt="o", ms=9, capsize=4,
                color=colors[k], label=k)
ax.set_xlabel("Region conservation (mean ± std over cell types)")
ax.set_ylabel("Cell type preservation")
ax.set_title(f"Tonsil, {PANEL} ({adata.n_vars:,} genes)")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
plt.tight_layout()
for ext in ("png", "svg"):
    plt.savefig(f"{OUT}region_celltype_scatter_{PANEL}.{ext}", bbox_inches="tight", dpi=300)
print("wrote", OUT)
