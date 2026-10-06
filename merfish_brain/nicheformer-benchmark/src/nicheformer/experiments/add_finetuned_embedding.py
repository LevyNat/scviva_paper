"""Copy the fine-tuned Nicheformer embedding into the MERFISH data file as obsm['X_nicheformer_e5'] (Fig. 2B-C).

The fine-tuned model is pretraining_fine_tune.py's epoch-04 checkpoint (the 5th epoch, counted from 0), and
get_embeddings.py run on it writes data_with_embeddings_epoch=04.h5ad with obsm['X_niche_embeddings'].
get_embeddings.ipynb (cells 47-48) copies the zero-shot embedding into the data file as 'X_nicheformer' the same
way; this script does it for the fine-tuned one.

    python add_finetuned_embedding.py            # writes the key if missing, checks it if present
"""
import sys

import anndata as ad
import h5py
import numpy as np

EMBEDDINGS = "/home/nathanl/Data/adata_M1_M2_core_6_sections_nicheformer_finetuned_epoch04.h5ad"
DATA = "/home/nathanl/Data/adata_M1_M2_core_6_sections.h5ad"
KEY = "X_nicheformer_e5"

with h5py.File(EMBEDDINGS, "r") as f:
    emb_names = f["obs"][f["obs"].attrs["_index"]][:].astype(str)
    emb = f["obsm"]["X_niche_embeddings"][:]

with h5py.File(DATA, "r") as f:
    names = f["obs"][f["obs"].attrs["_index"]][:].astype(str)
    stored = f["obsm"][KEY][:] if KEY in f["obsm"] else None

if len(names) != len(emb_names) or not (names == emb_names).all():
    sys.exit("cell order differs between the embedding file and the data file")

if stored is not None:
    diff = np.abs(stored - emb).max()
    print(f"{KEY} already in {DATA}: max |diff| vs epoch-04 embedding = {diff}")
    sys.exit(0 if diff == 0 else 1)

adata = ad.read_h5ad(DATA)
adata.obsm[KEY] = emb
adata.write_h5ad(DATA)
print(f"wrote {KEY} {emb.shape} into {DATA}")
