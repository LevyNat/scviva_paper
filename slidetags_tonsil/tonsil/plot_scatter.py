"""Region-vs-cell-type scatter for one panel, optionally leaving methods out of the view.

    python plot_scatter.py 15000hvg --drop PCA

Reads the summary written by score_panel.py; the CSVs themselves are never modified.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"
ap = argparse.ArgumentParser()
ap.add_argument("panel")
ap.add_argument("--drop", nargs="*", default=[])
ap.add_argument("--subdir", default="", help="variant subfolder of figures/tonsil_<panel>/")
a = ap.parse_args()

OUT = os.path.join(f"/home/nathanl/scviva_paper/slidetags_tonsil/figures/tonsil_{a.panel}", a.subdir) + "/"
s = pd.read_csv(f"{OUT}region_celltype_summary_{a.panel}.csv", index_col=0).drop(index=a.drop, errors="ignore")
COLORS = {"PCA": "#999999", "BANKSY (lambda 0.2)": "#377eb8", "SimVI": "#ff7f00", "scVI": "#4daf4a",
          "scVIVA (unlabeled)": "#fb9a99", "scANVI": "#984ea3", "scVIVA": "#e41a1c"}
fig, ax = plt.subplots(figsize=(6.5, 5))
for k, r in s.iterrows():
    ax.errorbar(r.region_cons, r.celltype_pres, xerr=r.region_std, fmt="o", ms=9, capsize=4,
                color=COLORS.get(k, "k"), label=k)
ax.set_xlabel("Region conservation (mean ± std over cell types)")
ax.set_ylabel("Cell type preservation")
ax.set_title(f"Tonsil, {a.panel}")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
plt.tight_layout()
tag = ("_no" + "_".join(d.replace(" ", "") for d in a.drop)) if a.drop else ""
for ext in ("png", "svg"):
    plt.savefig(f"{OUT}region_celltype_scatter_{a.panel}{tag}.{ext}", bbox_inches="tight", dpi=300)
print(s.round(3).to_string()); print("wrote", f"{OUT}region_celltype_scatter_{a.panel}{tag}.{{png,svg}}")
