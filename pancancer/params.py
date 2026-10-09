from dataclasses import dataclass, field

@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    # DATA_FOLDER: str = "/home/nathanlevy/Documents/exchange/"
    # DATA_FOLDER: str = "resolvae/"
    DATA_FOLDER: str = (
        "/home/nathanl/niche-VI-experiments/checkpoints/pancancer_resolvi/"
    )

    FIGURES_FOLDER: str = "figures_journal/"

    DATA_FILE: str = "pancancer_resolvi.h5ad"
    # for the data
    COORDS: str = "spatial"
    BATCH: str = "patient"
    SAMPLE: str = "patient"
    CELL_TYPE: str = "predicted_celltype"
    # CELL_TYPE: str = "predicted_celltype_coarse"
    NICHE: str = "niche"

    # for scVI
    EXPRESSION_MODEL = "scanvi"
    N_LAYERS: int = 1
    N_LATENT: int = 10
    LIKELIHOOD: str = "poisson"
    ######################
    N_EPOCHS_SCVI: int = 1000
    N_EPOCHS_RESOLVI: int = 500
    LR_SCVI: float = 1e-4
    BATCH_SIZE_SCVI: int = 4096
    OPTIMIZER: str = "Adam"
    ######################
    WEIGHT_DECAY: float = 1e-6
    KL_WARMUP: int | None = 400
    N_STEPS_KL_WARMUP: int | None = None
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
    BATCH_SIZE_NICHEVI: int = 4096
    REDUCE_LR_ON_PLATEAU: bool = True
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for runner control
    FORCE_RETRAIN: bool = False
    COMPUTE_UMAP: bool = True
    COMPUTE_SCIB: bool = True

    # QC (defaults match runner.py's getattr fallback)
    COUNTS_LAYER_FROM_RAW: bool = True
    MIN_COUNTS_PER_CELL: int | None = None

    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0
    MAX_SPATIAL_WEIGHT: float = 1.0

    # for nichetype scib
    TRESHOLD: int = 400

    # for scib benchmarking
    SCIB_PROPORTION: float = 0.3
    INCLUDE_EXPRESSION_LATENT_IN_SCIB: bool = True
    EMBEDDING_OBSM_KEYS: list[str] = field(default_factory=list)


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
        "niche_expression": "scvi",  # not used for now as only resolvi is implemented
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "prior_mixture": False,
        "prior_mixture_k": 1,
    },
}
