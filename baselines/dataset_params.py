import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PARAMS:
    DATA_FOLDER: str
    DATA_FILE: str
    FIGURES_FOLDER: str
    CHECKPOINTS_FOLDER: str
    COORDS: str = "spatial"
    BATCH: str = "batch"
    SAMPLE: str = "batch"
    CELL_TYPE: str = "cell_type"
    NICHE: Optional[str] = None
    TRESHOLD: int = 400
    # BANKSY input: True = X rebuilt as normalize_total+log1p of layers["counts"]; False = X as stored
    BANKSY_X_FROM_COUNTS: bool = True


DATASETS: dict[str, PARAMS] = {
    "brain_merfish": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/",
        DATA_FILE="adata_M1_M2_core_6_sections.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/merfish_brain/figures/revisions_benchmark/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/merfish_brain/checkpoints/",
        BATCH="brain_section_label",
        SAMPLE="brain_section_label",
        CELL_TYPE="cell_type",
        NICHE="major_brain_region",
        # the published brain BANKSY embedding was computed on X as stored
        BANKSY_X_FROM_COUNTS=False,
    ),
    # fixed, pre-reduced HVG file so BANKSY and SimVI always train on the exact same
    # gene set (the full 6278-gene panel doesn't fit SimVI's GPU memory budget on our 24GB GPU)
    "cosmx_cortex_hvg5000": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/cosmx_cortex",
        DATA_FILE="cosmx_cortex_hvg5000.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/cosmx_brain/figures/cosmx_cortex_hvg5000/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/cosmx_brain/checkpoints/cosmx_cortex_hvg5000/",
        BATCH="batch",
        SAMPLE="batch",
        CELL_TYPE="cell_type",
        NICHE="region",
    ),
    # CosMx duplicate experiment: astro.2 astrocytes
    # duplicated from the upper cortical layers (L2/3) into the vessel / peri-vascular region
    "cosmx_cortex_astro.2_upper_vessel_replaced": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/cosmx_cortex",
        DATA_FILE="cosmx_cortex_astro.2_upper_vessel_replaced.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/cosmx_brain/figures/duplicate/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/cosmx_brain/checkpoints/cosmx_cortex_astro.2_upper_vessel_replaced/",
        BATCH="batch",
        SAMPLE="batch",
        CELL_TYPE="cell_type",
        NICHE="region",
    ),
    "crc_visium_hd": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/VisiumHD_CRC/",
        DATA_FILE="adata_legacy_hvg4k.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/crc_visium_hd/figures/benchmark/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/",
        BATCH="sample",
        SAMPLE="sample",
        CELL_TYPE="cell_type_coarse",
        NICHE="region_label",
    ),
    "slidetags_tonsil_10000hvg": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/slidetags/",
        DATA_FILE="tonsil_nl_10000hvg.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/slidetags_tonsil/figures/tonsil_10000hvg/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/slidetags_tonsil/checkpoints/tonsil_10000hvg/",
        BATCH="donor_id",
        SAMPLE="donor_id",
        CELL_TYPE="cell_type",
        NICHE="region",
    ),
    # CRC duplicate experiment (crc_visium_hd/duplicate_analysis.ipynb, crc_visium_hd/duplicate/params.py)
    "crc_visium_hd_macrophages_replaced": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/VisiumHD_CRC/",
        DATA_FILE="adata_Macrophages_replaced.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/crc_visium_hd/figures/duplicate/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/crc_visium_hd/checkpoints/",
        BATCH="sample",
        SAMPLE="sample",
        CELL_TYPE="cell_type_coarse",
        NICHE="region_label",
    ),
    # MERFISH brain duplicate experiment (merfish_brain/duplicate_region/params.py)
    "brain_merfish_astrocyte_replaced": PARAMS(
        DATA_FOLDER="/home/nathanl/Data/",
        DATA_FILE="adata_astrocyte_replaced.h5ad",
        FIGURES_FOLDER="/home/nathanl/scviva_paper/merfish_brain/figures/duplicate_region/",
        CHECKPOINTS_FOLDER="/home/nathanl/scviva_paper/merfish_brain/checkpoints/",
        BATCH="brain_section_label",
        SAMPLE="brain_section_label",
        CELL_TYPE="cell_type",
        NICHE="major_brain_region",
        # same brain object as "brain_merfish", so the same BANKSY input: X as stored.
        BANKSY_X_FROM_COUNTS=False,
    ),
}


def add_dataset_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        required=True,
        help="which dataset's PARAMS to load from dataset_params.DATASETS",
    )
    return parser


def resolve_dataset(dataset_name: str) -> PARAMS:
    return DATASETS[dataset_name]
