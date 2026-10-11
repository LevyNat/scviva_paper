from dataclasses import dataclass, field


@dataclass(frozen=False)
class PARAMS:
    "scANVI + nicheVI on the 10,000-HVG tonsil panel (built by slidetags_build.ipynb)"

    DATA_FOLDER: str = "/home/nathanl/Data/slidetags/"
    # tonsil_nl_10000hvg.h5ad = top-10k seurat_v3 HVGs, with `region` (slidetags_build.ipynb)
    # and the 10k BANKSY/SimVI embeddings (written by run_benchmark.sh slidetags_tonsil_10000hvg)
    DATA_FILE: str = "tonsil_nl_10000hvg.h5ad"

    FIGURES_FOLDER: str = "/home/nathanl/scviva_paper/slidetags_tonsil/figures/tonsil_10000hvg/"
    CHECKPOINT_FOLDER: str = "/home/nathanl/scviva_paper/slidetags_tonsil/checkpoints/tonsil_10000hvg"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "donor_id"
    SAMPLE: str = "donor_id"
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

    TRESHOLD: int = 30

    # this dataset's counts layer is already built (see slidetags_build.ipynb)
    COUNTS_LAYER_FROM_RAW: bool = False

    # region + 10k banksy/simvi embeddings are present in tonsil_nl_10000hvg.h5ad
    COMPUTE_SCIB: bool = True
    SCIB_PROPORTION: float = 1.0
    # harmony left out on purpose: single donor, so it is a no-op copy of banksy_pc_20_lambda_0.2
    EMBEDDING_OBSM_KEYS: list[str] = field(
        default_factory=lambda: ["banksy_pc_20_nonspatial", "banksy_pc_20_lambda_0.2", "simvi25_both_mae_50"]
    )


setup = PARAMS()

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
