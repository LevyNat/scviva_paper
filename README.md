# scVIVA paper: reproducibility

Code behind the figures of the scVIVA manuscript: training runners, baseline benchmarks and figure notebooks.
scVIVA (`nichevi`) is the [archived niche-VI repository](https://github.com/YosefLab/niche-VI); the model is now maintained in [scviva-tools](https://scviva-tools.org).

## Installation

Linux x86-64, CUDA 12.4. The environment files pin the exact versions used for the paper.

```bash
git clone <this repo> scviva_paper && cd scviva_paper
conda env create -f envs/scvi.yml               # "scvi": scVIVA, scANVI, scIB, BANKSY, Hotspot, notebooks
conda env create -f baselines/envs/simvi24.yml  # "simvi24": SimVI baseline only
conda activate scvi
export JAX_PLATFORMS=cpu                        # scIB's JAX on CPU; PyTorch models still use the GPU
```

Three packages are pinned to specific commits in `envs/scvi.yml`; keep these pins:
- `niche-VI` (import `nichevi`): `YosefLab/niche-VI@34a85af`, not on PyPI
- `scvi-tools`: development commit `3e275ff` (2025-11-20), between releases 1.4.0 and 1.4.1
- `scib-metrics`: 0.5.7

## Data

All datasets are public. Set `DATA_FOLDER` in each `params.py` to your copy.

| dataset | source | figures |
|---|---|---|
| MERFISH mouse brain (Zhang et al., *Nature* 2023) | [CELLxGENE](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09) | 2, 3, S1A-S5, S10-S14 |
| CosMx human frontal cortex (Bruker Spatial Biology) | [CosMx FFPE human frontal cortex](https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/human-frontal-cortex-ffpe-dataset/) | S9 |
| Visium HD colorectal cancer (Oliveira et al., *Nat. Genet.* 2025) | [10x Genomics](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4) | S1B, S6, S7 |
| Xenium human breast cancer (Janesick et al., *Nat. Commun.* 2023) | [10x Genomics](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast) | 4, S15, S16, Table 1 |
| Breast cancer scRNA-seq atlas (Wu et al., *Nat. Genet.* 2021) | [CELLxGENE](https://cellxgene.cziscience.com/collections/dea97145-f712-431c-a223-6b5f565f362a) | S16, Table 1 |
| Breast scRNA-seq "Global Atlas" (*NAR Genom. Bioinform.*) | [CELLxGENE](https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb) | S17 |
| Slide-tags human tonsil (Russell et al., *Nature* 2024) | [Single Cell Portal SCP2169](https://singlecell.broadinstitute.org/single_cell/study/SCP2169) | S8 |
| MERSCOPE pan-cancer (Vizgen FFPE immuno-oncology release) | [Vizgen](https://info.vizgen.com/merscope-ffpe-solution) | 5, S18 |

## How to run

To run the benchmarks:
1. Build the input (or use the provided processed file).
2. Run the baselines: `(cd baselines && ./run_benchmark.sh <dataset> <gpu>)` (BANKSY and SimVI; datasets are listed in `baselines/dataset_params.py`). They write their embeddings into the data file, so they run before step 3.
3. Run scANVI + scVIVA and the scIB evaluation: `python runner.py <dataset_folder>` (from the repo root; settings in `<dataset_folder>/params.py`).
4. Run the figure notebooks.

Each dataset README gives the exact commands, and which figure each script or notebook makes.

## Datasets and figures

| README | figures and tables |
|---|---|
| [`merfish_brain/`](merfish_brain/README.md) | Fig. 2, 3, S10-S14 |
| ↳ [`merfish_brain/duplicate_region/`](merfish_brain/duplicate_region/README.md) | Fig. 2D-F, S4 |
| ↳ [`merfish_brain/robustness_to_annotation_error/`](merfish_brain/robustness_to_annotation_error/README.md) | Fig. S1A |
| ↳ [`merfish_brain/unlabeled/`](merfish_brain/unlabeled/README.md) | Fig. S2 |
| ↳ [`merfish_brain/ablation/`](merfish_brain/ablation/README.md) | Fig. S3 |
| ↳ [`merfish_brain/synthetic_gene/`](merfish_brain/synthetic_gene/README.md) | Fig. S5 |
| [`crc_visium_hd/`](crc_visium_hd/README.md) | Fig. S6, S7 |
| ↳ [`crc_visium_hd/cell_type_robustness/robustness_to_annotation_error/`](crc_visium_hd/cell_type_robustness/robustness_to_annotation_error/README.md) | Fig. S1B |
| [`cosmx_brain/`](cosmx_brain/README.md) | Fig. S9 |
| [`breast/`](breast/README.md) | Fig. 4, S15, S16, S17, Table 1 |
| [`slidetags_tonsil/`](slidetags_tonsil/README.md) | Fig. S8 |
| [`pancancer/`](pancancer/README.md) | Fig. 5, S18 |

## Trained models

#TODO 