import scvi

import scanpy as sc
from rich import print
import anndata as ad
import torch
import os

scvi.settings.seed = 34
# scvi.settings.device = torch.device("cuda:1")


CKPTS_DIR = "/home/nathanl/Data/VisiumHD_CRC/resolvi_ckpts/"
os.makedirs(CKPTS_DIR, exist_ok=True)


# Load data
adata = sc.read_h5ad("/home/nathanl/Data/VisiumHD_CRC/adata_recovered.h5ad")
sc.pp.filter_cells(adata, min_counts=10)

adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()

print(adata)

n_hidden = 256
n_epochs = 150
is_trained = os.path.exists(os.path.join(CKPTS_DIR, f"resolvi_model_{n_hidden}_epochs{n_epochs}"))
if not is_trained:
    print("Training RESOLVI model...")
    scvi.external.RESOLVI.setup_anndata(adata, layer="counts")
    resolvi = scvi.external.RESOLVI(adata, semisupervised=False, downsample_counts=False, n_hidden=n_hidden)
    resolvi.train(
        max_epochs=n_epochs,
        batch_size=128,
        # train_size=0.8,
        # validation_size=0,
        # early_stopping=True,
        # check_val_every_n_epoch=1,
    )
    resolvi.save(os.path.join(CKPTS_DIR, f"resolvi_model_{n_hidden}_epochs{n_epochs}"), overwrite=True)
else:
    print("Loading trained RESOLVI model...")
    resolvi = scvi.external.RESOLVI.load(
        os.path.join(CKPTS_DIR, f"resolvi_model_{n_hidden}_epochs{n_epochs}"), adata=adata
    )

adata.obsm[f"X_resolvi_{n_hidden}_epochs{n_epochs}"] = resolvi.get_latent_representation()

sc.pp.neighbors(adata, use_rep=f"X_resolvi_{n_hidden}_epochs{n_epochs}")
# sc.tl.leiden(adata, key_added=f"resolvi_leiden_{n_hidden}_epochs{n_epochs}", resolution=1.0)
sc.tl.leiden(
    adata,
    key_added=f"resolvi_leiden_{n_hidden}_epochs{n_epochs}",
    resolution=1.0,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)
print("Leiden done.")
sc.tl.umap(adata)
print("UMAP done.")
sc.pl.umap(
    adata,
    color=[f"resolvi_leiden_{n_hidden}_epochs{n_epochs}"],
    wspace=0.4,
    save=f"_resolvi_leiden_{n_hidden}_epochs{n_epochs}.png",
)

adata.write_h5ad("/home/nathanl/Data/VisiumHD_CRC/adata_resolvi.h5ad")
