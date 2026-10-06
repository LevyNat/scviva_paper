import os
from dataclasses import dataclass, field

# which duplication run to train on - set DUP_CELL_TYPE to switch cell types, e.g.
#   DUP_CELL_TYPE=oligodendrocyte python runner.py cosmx_brain/duplicate
DUP_CELL_TYPE = os.environ.get("DUP_CELL_TYPE", "endothelial")


@dataclass(frozen=False)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanl/Data/cosmx_cortex"

    FIGURES_FOLDER: str = "/home/nathanl/scviva_paper/cosmx_brain/figures/duplicate/"
    CHECKPOINT_FOLDER: str = "/home/nathanl/scviva_paper/cosmx_brain/checkpoints"

    # built by brain/cosmx_brain/duplicate_analysis.ipynb (see DUP_CELL_TYPE above)
    DATA_FILE: str = f"cosmx_cortex_{DUP_CELL_TYPE}_replaced.h5ad"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "batch"
    SAMPLE: str = "batch"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "region"

    # for scVI
    EXPRESSION_MODEL = "scanvi"
    N_LAYERS: int = 1
    N_LATENT: int = 10
    LIKELIHOOD: str = "poisson"
    ######################
    N_EPOCHS_SCVI: int = 1000
    LR_SCVI: float = 1e-4
    BATCH_SIZE_SCVI: int = 1024
    OPTIMIZER: str = "Adam"
    ######################
    WEIGHT_DECAY: float = 1e-6
    KL_WARMUP: int | None = 400
    MAX_KL_WEIGHT: float = 1

    # for nicheVI
    K_NN: int = 20
    N_LAYERS_NICHE: int = 1
    N_LAYERS_COMPO: int = 1
    N_HIDDEN: int = 128
    N_HIDDEN_COMPO: int = 128
    N_HIDDEN_NICHE: int = 128
    N_LATENT_NICHEVI: int = 10
    ######################
    N_EPOCHS_NICHEVI: int = 1000
    LR_NICHEVI: float = 5e-4
    BATCH_SIZE_NICHEVI: int = 1024
    REDUCE_LR_ON_PLATEAU: bool = True
    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0 if SPATIAL_WARMUP is None else 0
    MAX_SPATIAL_WEIGHT: float = 1.0
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for runner control
    # this run only needs the trained embeddings + nicheVI niche/compo errors for the
    # duplication analysis - the scib benchmark and runner UMAPs would cost hours for nothing
    FORCE_RETRAIN: bool = False
    COMPUTE_UMAP: bool = False
    COMPUTE_SCIB: bool = False

    # QC (defaults match runner.py's getattr fallback)
    COUNTS_LAYER_FROM_RAW: bool = False
    MIN_COUNTS_PER_CELL: int | None = None

    # for nichetype scib
    TRESHOLD: int = 400

    # for scib benchmarking
    SCIB_PROPORTION: float = 0.3
    INCLUDE_EXPRESSION_LATENT_IN_SCIB: bool = True
    EMBEDDING_OBSM_KEYS: list[str] = field(
        default_factory=lambda: [
            "banksy_pc_20_lambda_0.2",
            "simvi25_both_mae_50",
        ]
    )


setup = PARAMS()

# ----- NicheVI settings -----#
niche_setup = {
    f"s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 10,
        "compo_rec_weight": 10,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": setup.N_LAYERS_NICHE,
        "n_layers_compo": setup.N_LAYERS_COMPO,
        "n_hidden_niche": setup.N_HIDDEN_NICHE,
        "n_hidden_compo": setup.N_HIDDEN_COMPO,
        "niche_expression": "scvi",
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "prior_mixture": False,
        "prior_mixture_k": 1,
    },
}
