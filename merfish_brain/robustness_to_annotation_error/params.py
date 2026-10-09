from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanl/Data/"

    FIGURES_FOLDER: str = "/home/nathanl/scviva_paper/merfish_brain/figures/robustness_to_annotation_error/seeds/"

    DATA_FILE: str = "adata_M1_M2_core_6_sections.h5ad"
    # name of the checkpoint folder and of the output file
    RUN_NAME: str = "adata_M1_M2_core_6_sections_seed"

    SEED: int = 34  # Default seed (kept for backwards compatibility)

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "brain_section_label"
    SAMPLE: str = "brain_section_label"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "major_brain_region"

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
    N_EPOCHS_NICHEVI: int = 1000
    LR_NICHEVI: float = 5e-4
    BATCH_SIZE_NICHEVI: int = 1024
    REDUCE_LR_ON_PLATEAU: bool = True
    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0 if SPATIAL_WARMUP is None else 0
    MAX_SPATIAL_WEIGHT: float = 1.0
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for nichetype scib
    TRESHOLD: int = 200


setup = PARAMS()

# ----- Seeds for multi-seed experiments -----#
SEEDS_TO_TEST = [34, 42, 123]


# ----- Base NicheVI configuration -----#
def make_niche_config(
    label_to_flip: float,
    seed: int,
    spatial_weight: float = 10,
    niche_rec_weight: float = 10,
    compo_rec_weight: float = 10,
) -> dict:
    """Create a NicheVI configuration with specified flip rate and seed."""
    return {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": spatial_weight,
        "niche_rec_weight": niche_rec_weight,
        "compo_rec_weight": compo_rec_weight,
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
        "label_to_flip": label_to_flip,
        "seed": seed,  # Per-configuration seed
    }


# ----- NicheVI settings -----#
# Generate configurations for f0, f1, f5, f10 with multiple seeds
niche_setup = {}

# Flip rates to test with multiple seeds
FLIP_RATES = {
    "f0": 0.0,
    "f1": 0.01,
    "f5": 0.05,
    "f10": 0.10,
}

# Generate configurations for all flip rates with all seeds
for flip_name, flip_rate in FLIP_RATES.items():
    for seed in SEEDS_TO_TEST:
        key = f"{flip_name}_s10_{setup.EXPRESSION_MODEL}_lr{setup.LR_NICHEVI}_{setup.LIKELIHOOD}_seed{seed}"
        niche_setup[key] = make_niche_config(flip_rate, seed)
