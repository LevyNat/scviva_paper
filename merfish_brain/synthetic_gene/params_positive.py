from dataclasses import dataclass

import seaborn as sns


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanl/Data/"
    # DATA_FOLDER: str = "/home/labs/nyosef/Collaboration/SpatialDatasets/merfish/whole_brain_atlas_zhuang/"

    FIGURES_FOLDER: str = "/home/nathanl/scviva_paper/merfish_brain/figures/"

    FIGURES_FOLDER_SCANVI = FIGURES_FOLDER + "scanvi/"
    FIGURES_FOLDER_RESOLVI = FIGURES_FOLDER + "resolvi/"
    FIGURES_FOLDER_NICHEVI = FIGURES_FOLDER + "nichevi/"
    FIGURES_FOLDER_POSITIVE_SHUFFLE = FIGURES_FOLDER + "positive_shuffle/"
    FIGURES_FOLDER_SYNTHETIC_GENES = FIGURES_FOLDER + "synthetic_genes/"

    # DATA_FILE: str = "adata_M1_M2_core_6_sections.h5ad"
    DATA_FILE: str = "adata_M1_M2_core_6_sections_with_synthetic_gene.h5ad"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "brain_section_label"
    SAMPLE: str = "brain_section_label"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "major_brain_region"
    # for positive shuffle
    N_SYNTHETIC_GENES: int = 10
    SIGNAL_BOUNDS: tuple = (3.0, 3.5)

    # for scVI
    EXPRESSION_MODEL = "scanvi"
    N_LAYERS: int = 1
    N_LATENT: int = 10
    LIKELIHOOD: str = "poisson"
    ######################
    N_EPOCHS_SCVI: int = 1000
    N_EPOCHS_RESOLVI: int = 100
    LR_SCVI: float = 1e-4
    BATCH_SIZE_SCVI: int = 1024
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
    N_HEADS: int | None = None
    N_TOKENS_DECODER: int | None = None
    ######################
    N_EPOCHS_NICHEVI: int = 1001
    LR_NICHEVI: float = 5e-4
    BATCH_SIZE_NICHEVI: int = 1024
    REDUCE_LR_ON_PLATEAU: bool = True
    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0 if SPATIAL_WARMUP is None else 0
    MAX_SPATIAL_WEIGHT: float = 1.0
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for nichetype scib
    TRESHOLD: int = 400


setup = PARAMS()

# ----- NicheVI settings -----#
niche_setup = {
    f"1g_eta0_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 0,  # no niche reconstruction loss
        "compo_rec_weight": 10,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": setup.N_LAYERS_NICHE,
        "n_layers_compo": setup.N_LAYERS_COMPO,
        "n_hidden_niche": setup.N_HIDDEN_NICHE,
        "n_hidden_compo": setup.N_HIDDEN_COMPO,
        "niche_expression": "resolvi",
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "n_heads": None,
        "n_tokens_decoder": setup.N_TOKENS_DECODER,
        "prior_mixture": False,
        "prior_mixture_k": 1,
        "add_synthetic_type": False,
    },
    f"1g_synth_eta0_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 0,  # no niche reconstruction loss
        "compo_rec_weight": 10,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": setup.N_LAYERS_NICHE,
        "n_layers_compo": setup.N_LAYERS_COMPO,
        "n_hidden_niche": setup.N_HIDDEN_NICHE,
        "n_hidden_compo": setup.N_HIDDEN_COMPO,
        "niche_expression": "resolvi",
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "n_heads": None,
        "n_tokens_decoder": setup.N_TOKENS_DECODER,
        "prior_mixture": False,
        "prior_mixture_k": 1,
        "add_synthetic_type": True,
    },
}
