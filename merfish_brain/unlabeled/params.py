# Fig. S2: scVIVA trained on Leiden pseudo-labels instead of cell types.
# Run:  cd merfish_brain/unlabeled && python runner_no_labels.py   (legacy env)
# Copy of merfish_brain/ablation/params.py without its niche_setup: runner_no_labels.py builds one
# config per Leiden resolution itself and switches EXPRESSION_MODEL to scvi
from dataclasses import dataclass, field


@dataclass(frozen=False)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanl/Data/"
    # DATA_FOLDER: str = "/home/labs/nyosef/Collaboration/SpatialDatasets/merfish/whole_brain_atlas_zhuang/"

    FIGURES_FOLDER: str = "/home/nathanl/scviva_paper/merfish_brain/figures/revisions_benchmark/"
    CHECKPOINT_FOLDER: str = "/home/nathanl/scviva_paper/merfish_brain/checkpoints"

    FIGURES_FOLDER_SHUFFLE = FIGURES_FOLDER + "cLISI/"

    DATA_FILE: str = "adata_M1_M2_core_6_sections.h5ad"
    # DATA_FILE: str = "adata_M1_M2_core_6_sections_simvi25_shuffled.h5ad"

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
    N_EPOCHS_RESOLVI: int = 500
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
    FORCE_RETRAIN: bool = False
    COMPUTE_UMAP: bool = False
    COMPUTE_SCIB: bool = True

    # QC (defaults match runner.py's getattr fallback)
    COUNTS_LAYER_FROM_RAW: bool = True
    MIN_COUNTS_PER_CELL: int | None = None

    # for nichetype scib
    TRESHOLD: int = 400

    # Leiden resolutions on the scVI latent used as pseudo-labels (one scVIVA model each)
    LEIDEN_RES: tuple = (0.1, 0.15, 0.2, 0.3, 0.35)
    # resolutions shown in Fig. S2 (region conservation per Leiden cluster, UMAP): 0.15 gives 17 clusters, the number
    # of annotated cell types; 0.35 maximises the silhouette of the Leiden clusters on the scVI latent (select_resolution.py)
    FIGURE_LEIDEN_RES: tuple = (0.15, 0.35)

    # for scib benchmarking
    SCIB_PROPORTION: float = 0.3
    INCLUDE_EXPRESSION_LATENT_IN_SCIB: bool = False
    # outputs <stem>_ablation_k_nicheVI.h5ad and *_<stem>_ablation_k.csv, so this run does not
    # overwrite the Figure_2 outputs that share the same checkpoint folder
    OUTPUT_TAG: str = "_ablation_k"
    # extra precomputed obsm keys to compare against nicheVI in scib (e.g. "banksy_pc_20_lambda_0.2", "X_banksy_02_harmony", "X_nicheformer", "simvi25_both_mae_50")
    EMBEDDING_OBSM_KEYS: list[str] = field(default_factory=list)


setup = PARAMS()

# ----- NicheVI settings -----#
