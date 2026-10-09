from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    # DATA_FOLDER: str = (
    #     "/home/labs/nyosef/Collaboration/SpatialDatasets/xenium/human_breast_cancer/"
    # )
    DATA_FOLDER: str = "/home/nathanl/Data/"
    # DATA_FOLDER: str = "/home/nathanlevy/sda/Data/"
    DATA_FILE: str = "xenium_breast_cancer_S1_R1_2.h5ad"

    FIGURES_FOLDER: str = "figures_journal/"
    FIGURES_2DE_FOLDER: str = "/home/nathanl/scviva_paper/breast/differential_expression_benchmark/figures_2DE/"
    # for the data
    COORDS: str = "spatial"
    BATCH: str = "sample"
    SAMPLE: str = "sample"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "kmeans_alpha_5"

    # for scVI
    EXPRESSION_MODEL = "scanvi"
    N_LAYERS: int = 1
    N_LATENT: int = 10
    LIKELIHOOD: str = "poisson"
    ######################
    N_EPOCHS_SCVI: int = 1001
    LR_SCVI: float = 1e-4
    BATCH_SIZE_SCVI: int = 512
    OPTIMIZER: str = "Adam"
    ######################
    WEIGHT_DECAY: float = 1e-6
    KL_WARMUP: Optional[int] = 400
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
    # Two scVIVA checkpoints are used: the runner computes the embedding with the 1001-epoch model, and the
    # lvm-DE of Fig. 4E and Table 1 (hotspot_breast.ipynb, t_test_xenium.ipynb) uses the 1000-epoch model
    N_EPOCHS_NICHEVI: int = 1001
    N_EPOCHS_NICHEVI_DE: int = 1000
    LR_NICHEVI: float = 5e-4
    BATCH_SIZE_NICHEVI: int = 1024
    REDUCE_LR_ON_PLATEAU: bool = True
    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0 if SPATIAL_WARMUP is None else 0
    MAX_SPATIAL_WEIGHT: float = 1.0
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for runner control
    FORCE_RETRAIN: bool = False
    COMPUTE_UMAP: bool = True
    COMPUTE_SCIB: bool = True

    # QC (defaults match runner.py's getattr fallback)
    COUNTS_LAYER_FROM_RAW: bool = True
    MIN_COUNTS_PER_CELL: int | None = None

    # for nichetype scib
    TRESHOLD: int = 400

    # for scib benchmarking
    SCIB_PROPORTION: float = 0.3
    INCLUDE_EXPRESSION_LATENT_IN_SCIB: bool = True
    EMBEDDING_OBSM_KEYS: list[str] = field(default_factory=list)


setup = PARAMS()

# ----- NicheVI settings -----#
niche_setup = {
    f"ln_s10_w400_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 10,
        "compo_rec_weight": 10,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": 1,
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
