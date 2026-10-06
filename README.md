# scVIVA paper: reproducibility

This repo has the code behind the scVIVA manuscript figures: training runners, baseline benchmarks and figure notebooks. `not_in_paper/` holds experiments the manuscript doesn't use (the old k-sweep folder, naive concatenations, seeded runs, breast per-region scib, the breast BANKSY shuffle script, the tonsil duplicate, and brain runners replaced by the universal runner).

scVIVA (`nichevi`) is the [archived niche-VI repository](https://github.com/YosefLab/niche-VI). The model is now maintained in [scvi-tools](https://scvi-tools.org).

## Installation

The environment files are exported from the machine that produced the paper results: Linux x86-64, CUDA 12.4. They pin exact conda builds, so they install on Linux only.

```bash
git clone <this repo> scviva_paper && cd scviva_paper
conda env create -f envs/scvi.yml               # env "scvi": scVIVA, scANVI, scib, BANKSY, Hotspot, notebooks
conda env create -f baselines/envs/simvi24.yml  # env "simvi24": SimVI baseline only
conda activate scvi
export JAX_PLATFORMS=cpu                        # see below
```

Notes:
- **JAX on CPU.** On our machine, the JAX CUDA plugin pinned in `scvi` failed to start ("Unable to initialize backend 'cuda'"). Everything in this repo was run with `JAX_PLATFORMS=cpu`. PyTorch, and with it every scANVI and scVIVA model, still uses the GPU.
- **BANKSY** is vendored in `baselines/BANKSY/` and needs no install.
- **The C-SIDE notebooks** for Table 1 (`breast/differential_expression_benchmark/c_side_*.ipynb`) run in an R 4.3.3 kernel with `spacexr`. No environment file is provided for them.
- **Nicheformer** (Fig. 2B baseline) was run in its own environment, on Python 3.11, following `merfish_brain/nicheformer-benchmark/README.md`. No environment file is provided for it either.

| env | used by | file |
|---|---|---|
| `scvi` | `runner.py`, every dataset runner and notebook, BANKSY | `envs/scvi.yml` |
| `simvi24` | SimVI (installed `simvi==0.1.2` from PyPI) | `baselines/envs/simvi24.yml` |

`envs/scvi.yml` installs every package at the version used for the paper. For three packages, the paper environment used local clones; the YAML pins the same code. Do not replace these pins with PyPI releases:

| package | pinned in `envs/scvi.yml` as | why |
|---|---|---|
| `niche-VI` (import `nichevi`) | `git+https://github.com/YosefLab/niche-VI.git@34a85af` | not on PyPI |
| `scvi-tools` | `git+https://github.com/scverse/scvi-tools.git@3e275ff` | a development commit on `main` (2025-11-20), between the 1.4.0 and 1.4.1 releases and identical to neither. Installed from git it reports `1.4.0.post1`. The paper env's editable clone reported `1.3.1`, a version string left over from an older build of the clone. |
| `scib-metrics` | `scib-metrics==0.5.7` | the clone was at `0404e63`, whose code is identical to the 0.5.7 release (only `.pre-commit-config.yaml` differs) |

None of the three clones had uncommitted changes.

## Data

Data is not in the repo. Every dataset is public. The params files read from `/home/nathanl/Data` unless they say otherwise; change `DATA_FOLDER` to your copy.

### Public sources

| dataset | publication | download | what the code reads | figures |
|---|---|---|---|---|
| MERFISH mouse brain (Zhuang lab, ~1,100 genes) | Zhang et al., *Nature* 624, 2023 | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09) | mice M1 and M2; slices `C57BL6J-1.074`, `-1.077`, `-1.079`, `-2.035`, `-2.036`, `-2.037`; author cell types and major brain regions | 2, 3, S1A-S5, S10-S14 |
| Slide-tags human tonsil | Russell et al., *Nature* 625, 2024 | [Single Cell Portal SCP2169](https://singlecell.broadinstitute.org/single_cell/study/SCP2169) | `HumanTonsil_expression.csv.gz`, `HumanTonsil_metadata.csv`, `HumanTonsil_spatial.csv`, `HumanTonsil_cluster.csv` | S8 |
| Visium HD colorectal cancer | Oliveira et al., *Nat. Genet.* 57, 2025 | [10x dataset, Space Ranger 4.0.1](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4); [10XGenomics/HumanColonCancer_VisiumHD](https://github.com/10XGenomics/HumanColonCancer_VisiumHD) for metadata | `segmented_outputs/` (`filtered_feature_cell_matrix.h5`, `cell_segmentations.geojson`), `spatial/`, `Visium_HD_Human_Colon_Cancer_barcode_mappings.parquet` | S1B, S6, S7 |
| CosMx human frontal cortex (6K panel) | Bruker Spatial Biology | [CosMx FFPE human frontal cortex](https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/human-frontal-cortex-ffpe-dataset/) | flat files `S3_exprMat_file.csv`, `S3_metadata_file.csv`, `S3_fov_positions_file.csv`, `S3-polygons.csv`, `S3_tx_file.csv.gz` | S9 |
| Xenium human breast cancer | Janesick et al., *Nat. Commun.* 14, 2023 | [10x Xenium FFPE human breast preview data](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast) | sample 1, replicates 1 and 2 (`S1_R1`, `S1_R2`): Xenium outputs and transcripts for Proseg | 4, S15, S16, Table 1 |
| breast cancer scRNA-seq atlas | Wu et al., *Nat. Genet.* 53, 2021 | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/dea97145-f712-431c-a223-6b5f565f362a) ([h5ad](https://datasets.cellxgene.cziscience.com/cbbb607f-578b-47ba-858b-407bb8be917f.h5ad)) | saved as `breast_sc_atlas_wu2021.h5ad`; the DE notebooks build the endothelial reference `breast_sc_atlas_wu2021_DE.h5ad` from it | S16, Table 1 |
| breast scRNA-seq "Global Atlas" | *NAR Genom. Bioinform.*, doi 10.1093/nargab/lqaf217 | [CELLxGENE collection](https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb) ([h5ad](https://datasets.cellxgene.cziscience.com/7cdea341-ca7a-40fd-8192-b8ecb2d7b91e.h5ad)) | saved as `breast_sc_atlas_global.h5ad` | S17 |
| MERSCOPE pan-cancer (FFPE immuno-oncology release, 8 cancer types) | Vizgen | [Vizgen MERSCOPE FFPE data release](https://info.vizgen.com/merscope-ffpe-solution) (request form) | all 8 tissues, re-segmented with Proseg | 5, S18 |
| Nicheformer weights (baseline only) | Schaar et al., 2024 (not in the manuscript's references) | pretrained checkpoint linked from [theislab/nicheformer](https://github.com/theislab/nicheformer) | `nicheformer.ckpt`, then fine-tuned with `merfish_brain/nicheformer-benchmark/` | 2B |

### Derived inputs the runners read

Each runner reads a processed h5ad built from those downloads. Where no script in this repo builds the file, the last column says so.

| file (under `DATA_FOLDER`) | read by | built by |
|---|---|---|
| `adata_M1_M2_core_6_sections.h5ad` | `merfish_brain/{Figure_2,ablation,unlabeled,robustness_to_annotation_error}`, baselines `brain_merfish` | not scripted here (subset of the CELLxGENE object, `raw.X` = counts) |
| `adata_astrocyte_replaced.h5ad` | `merfish_brain/duplicate_region` | `duplicate_region/duplicate_analysis.ipynb` |
| `adata_M1_M2_core_6_sections_with_synthetic_gene.h5ad` | `merfish_brain/synthetic_gene` | `synthetic_gene_playground.ipynb` |
| `slidetags/tonsil_nl_10000hvg.h5ad` | `slidetags_tonsil/tonsil` | `slidetags_build.ipynb`, `tonsil.ipynb` cell 20, `build_nl_panels.py 10000` |
| `VisiumHD_CRC/adata_legacy_hvg4k.h5ad` | `crc_visium_hd`, baselines `crc_visium_hd` | `crc_analysis.ipynb` + `resolvi.py` |
| `VisiumHD_CRC/adata_Macrophages_replaced.h5ad` | CRC duplicate | `crc_visium_hd/duplicate_analysis.ipynb` |
| `cosmx_cortex/cosmx_cortex_hvg5000.h5ad` | `cosmx_brain`, baselines `cosmx_cortex_hvg5000` | `cosmx_cortex.ipynb` cell 9 (seurat_v3, 5,000 HVGs on raw counts) |
| `cosmx_cortex/cosmx_cortex_astro.2_upper_vessel_replaced.h5ad` | `cosmx_brain/duplicate` | `cosmx_brain/duplicate_analysis.ipynb` |
| `xenium_breast_cancer_S1_R1_2.h5ad` | `breast/breast_original` | not scripted here |
| `proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad` | `breast/breast_proseg` | not reproducible from this repo: `breast/preprocessing/` documents how it was built (Proseg, scANVI and resolVI label transfer), but the intermediate models are not available |
| `pancancer_resolvi.h5ad` | `pancancer/` | not scripted here |

**Warning:** several scripts (the baselines, the CRC legacy runner, the CosMx notebook) write their results back into these data files. Snapshot `DATA_FOLDER` before re-running.

## Layout

Each dataset folder has its own `figures/` subfolder. `checkpoints/` folders are created by the runners and are gitignored.

| folder | contents |
|---|---|
| `runner.py` | universal runner: `python runner.py <dataset_dir>` reads `<dataset_dir>/params.py`, trains scANVI then scVIVA, and runs scib |
| `baselines/` | SimVI + BANKSY: `./run_benchmark.sh <dataset> <gpu>`; datasets are listed in `dataset_params.py` |
| `merfish_brain/` | Figure 2 benchmark, ablations and K/β sweeps, unlabeled run, label-noise robustness, astrocyte duplicate, synthetic gene, Hotspot, Nicheformer |
| `cosmx_brain/` | CosMx cortex preprocessing, benchmark and cell-duplication experiments |
| `crc_visium_hd/` | preprocessing (resolVI + Leiden annotation), benchmark, macrophage duplicate, memory sweep, label-noise robustness |
| `breast/` | Xenium original and proseg runs, Hotspot, per-region scib, DE benchmark (t-test, C-SIDE), proseg/resolVI preprocessing |
| `pancancer/` | pan-cancer analysis; does not use the universal runner |
| `slidetags_tonsil/` | data build and the 10k-HVG benchmark, with labelled and unlabeled runs |

## Run order

For every dataset, the steps are:
1. Build the input.
2. Run the baselines, because they write their embeddings into the data file.
3. Run scANVI + scVIVA with `runner.py`.
4. Run the figure notebook.

Commands below are run from the repo root.

**MERFISH brain**
```bash
(cd baselines && ./run_benchmark.sh brain_merfish 0)
python runner.py merfish_brain/Figure_2                    # main benchmark (Figure_2/figures/)
python runner.py merfish_brain/ablation                     # Fig. S3: ablations s0/eta0 and K/β sweeps -> figures/revisions_benchmark/*_ablation_k.csv
(cd merfish_brain/unlabeled && python runner_no_labels.py)  # Fig. S2: unlabeled scVIVA -> figures/revisions_benchmark/*_ablation_nolabel.csv
(cd merfish_brain/robustness_to_annotation_error && python runner.py)   # Fig. S1A
# Fig. S5: merfish_brain/synthetic_gene/synthetic_gene_playground.ipynb cells 21-29 -> runner_synthetic_gene.py -> cells 42-48
# notebooks: merfish_brain/Figure_2_BC.ipynb (run the cells in order), hotspot_astro_benchmark.ipynb
# fine-tuned Nicheformer embedding (Fig. 2B-C): merfish_brain/nicheformer-benchmark/src/nicheformer/experiments/add_finetuned_embedding.py
```
Astrocyte duplicate (Fig. 2D-F, S4): `(cd baselines && ./run_benchmark.sh brain_merfish_astrocyte_replaced 0)`, then `python runner.py merfish_brain/duplicate_region`, then `merfish_brain/duplicate_region/duplicate_analysis.ipynb`.

Each brain experiment has its own params folder; `Figure_2/`, `ablation/` and `duplicate_region/` run through the universal `runner.py`. Only the unlabeled, annotation-noise and synthetic-gene experiments keep a dedicated runner, because they do something the universal runner can't (Leiden pseudo-labels, label flipping with seeds, a synthetic cell type).

**CosMx cortex**
```bash
# cosmx_brain/cosmx_cortex.ipynb builds the data, the 5000-HVG file (cell 9) and the `region` labels
(cd baselines && ./run_benchmark.sh cosmx_cortex_hvg5000 0)
python runner.py cosmx_brain
# duplicate (Fig. S9D-F uses run_tag astro.2_upper_vessel): build it in cosmx_brain/duplicate_analysis.ipynb, then
(cd baselines && ./run_benchmark.sh cosmx_cortex_astro.2_upper_vessel_replaced 0)
DUP_CELL_TYPE=astro.2_upper_vessel python runner.py cosmx_brain/duplicate
```

**CRC Visium HD**
```bash
# crc_visium_hd/crc_analysis.ipynb (segmentation -> HVG) -> crc_visium_hd/resolvi.py -> back in the notebook for annotation and regions
(cd baselines && ./run_benchmark.sh crc_visium_hd 0)
(cd crc_visium_hd && python runner.py)    # legacy runner, writes into the data file. The notebooks read that file.
# notebooks: crc_analysis.ipynb (figures), cell_type_DE.ipynb, duplicate_analysis.ipynb
(cd crc_visium_hd && python memory_sweep.py)    # plus baselines/SIMVI/memory_sweep.py and baselines/BANKSY/benchmark_memory.py
```

**Breast** (no BANKSY/SimVI arm): run the dedicated runners in `breast/breast_original/` and `breast/breast_proseg/`, from inside each folder. The universal runner cannot run breast. They load the included models and write the `_nicheVI.h5ad` files next to them; the Hotspot and DE notebooks read those files. Then run the Hotspot and DE notebooks, and the C-SIDE R notebooks for Table 1.

**Pancancer:** the notebooks are provided as run for the paper; their intermediate files are not built by scripts in this repo, so they cannot be re-run from it alone.

**Slide-tags tonsil**
```bash
# slidetags_tonsil/slidetags_build.ipynb, then tonsil/tonsil.ipynb cell 20 for the region labels
(cd slidetags_tonsil/tonsil && python build_nl_panels.py 10000)
slidetags_tonsil/tonsil/run_panel.sh 10000hvg     # unlabeled arm + baselines + runner + scoring
# notebooks: tonsil/tonsil_10000hvg.ipynb, tonsil/tonsil_10000hvg_grid.ipynb
```

## Checkpoints

The runners load a trained model when its folder exists in the experiment's `checkpoints/` folder, so the figures reproduce without retraining; delete a model folder to retrain that model. `checkpoints/` is gitignored: the models are not in git.

Each model folder holds only `model.pt`, except the two Fig. S5 synthetic-gene folders, which also keep the `adata.h5ad` that `synthetic_gene_playground.ipynb` cells 42 and 44 read (7.5 GB of the 8.8 GB).

SimVI always retrains: its runner does not load existing checkpoints.

## Figures → code

The panel mapping below comes from the manuscript's legends.

| figure | code |
|---|---|
| Fig. 1 | schematic, no code |
| Fig. 2A-C | `merfish_brain/Figure_2_BC.ipynb`, using `runner.py merfish_brain/Figure_2`, the baselines (`brain_merfish`) and `nicheformer-benchmark/` |
| Fig. 2D-F, S4 | `merfish_brain/duplicate_region/` (astrocytes from thalamus/hypothalamus duplicated into the isocortex) |
| Fig. 3, S10-S14 | `merfish_brain/hotspot_astro_benchmark.ipynb` |
| Fig. 4, S15-S16 | `breast/breast_original/hotspot_breast.ipynb`, `breast/breast_proseg/hotspot_breast.ipynb`, `breast/differential_expression_benchmark/t_test_*.ipynb`; Proseg/resolVI in `breast/preprocessing/` |
| Table 1 | `breast/differential_expression_benchmark/benchmark_2DE.ipynb`, `c_side_xenium*.ipynb`, `t_test_*.ipynb` |
| Fig. S17 | `breast/breast_original/breast_scrna_DE.ipynb` |
| Fig. 5, S18 | `pancancer/` (`T_cell_filtering.ipynb`, `filtered_T_cell_analysis.ipynb`, `plot_and_process_dataset.ipynb`, `hotspot_runner.py`) |
| Fig. S1A | `merfish_brain/robustness_to_annotation_error/` |
| Fig. S1B | `crc_visium_hd/cell_type_robustness/robustness_to_annotation_error/` |
| Fig. S2 | `merfish_brain/unlabeled/runner_no_labels.py`, then `Figure_2_BC.ipynb` cells 40-43 |
| Fig. S3 | `runner.py merfish_brain/ablation`, then `Figure_2_BC.ipynb` cells 30 and 33-34 |
| Fig. S5 | `merfish_brain/synthetic_gene/` |
| Fig. S6A-C, S7 | `crc_visium_hd/crc_analysis.ipynb`, `resolvi.py`, `cell_type_DE.ipynb` |
| Fig. S6D | `crc_visium_hd/memory_sweep.py`, `baselines/SIMVI/memory_sweep.py`, `baselines/BANKSY/benchmark_memory.py` |
| Fig. S6E-G | `crc_visium_hd/duplicate_analysis.ipynb` |
| Fig. S8 | `slidetags_tonsil/` (`tonsil/run_panel.sh 10000hvg`, `tonsil_10000hvg.ipynb`, `tonsil_10000hvg_grid.ipynb`) |
| Fig. S9 | `cosmx_brain/` (`cosmx_cortex.ipynb`, `python runner.py cosmx_brain`, `duplicate_analysis.ipynb` with `astro.2_upper_vessel`) |
