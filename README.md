[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13?style=for-the-badge)](https://openreview.net/forum?id=ey4IT8N8rl)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2026-4b44ce?style=for-the-badge)](https://openreview.net/forum?id=ey4IT8N8rl)

# Preserving Geometric Symmetry in Uncertainty Estimation for Molecular Forces

This repository contains the code for the NeurIPS 2026 paper
**"Preserving Geometric Symmetry in Uncertainty Estimation for Molecular Forces"**.
This `main` branch covers the MD17 force experiments; the QM7X dipole
experiments are on the `qm7x` branch.

The project studies uncertainty estimation for equivariant vector-valued molecular
properties such as atomic forces and dipole moments. The main method is
Equivariant Multivariate Evidential Regression (EMER), which parameterizes the
Normal-Inverse-Wishart scale matrix as

```math
\boldsymbol{\Psi}(\alpha,\mathbf{F})=\exp(\alpha)\,\mathbf{I}+\mathbf{F}\mathbf{F}^{\top}
```

where `alpha` is an invariant scalar and `F` is an equivariant low-rank vector
feature matrix. This construction keeps the predicted uncertainty covariance
positive definite and equivariant under 3D rotations.

<img src="figures/equivariant_uncertainty.png" width="100%" align="center">

For a rotation `R`, the predicted distribution transforms as `R mu` and `R Sigma R^T`, i.e. the uncertainty rotates together with the predicted vector.

## Framework

<img src="figures/framework.png" width="70%" align="center">

On top of the vector head that predicts the target, two uncertainty heads output the NIW parameters: a scalar head for `nu` and `alpha`, and a vector head for `F`. The network is trained end-to-end with the evidential negative log-likelihood. With `-D` variants, dropout is active during training only and acts as a regularizer.

## What is included

- EMER and MER heads, with and without dropout (`-D`), for PaiNN-style equivariant
  force prediction.
- Multivariate evidential loss for vector-valued targets.
- Mahalanobis distance-based calibration metrics for vector uncertainty.
- MD17 and MD17-RotOOD dataset loaders, including planar alignment and controlled
  rotation shift.
- Training and evaluation entry point based on PyTorch Lightning.
- Example configs for MD17-RotOOD backbone and EMER experiments.
- A minimal vendored subset of `ocpmodels` from the Open Catalyst Project
  (`BaseModel`, dataset loader and normalizer).

## Repository layout

```text
.
|-- parallel_lightning.py                 # Main train/evaluate entry point
|-- training_lightning.py                 # Lightning training modules
|-- ue.yml                                # Conda environment
|-- configs/
|   `-- aspirin/                          # PaiNN backbone, MER, and EMER configs
|-- uncertainty_ai/
|   |-- dataset/                          # MD17, MD17Aligned, QM7X loaders
|   |-- loss/                             # Scalar and vector evidential losses
|   |-- metrics/                          # Calibration and NLL metrics
|   `-- models/                           # PaiNN, MACE-style, and dropout models
|-- ocpmodels/                            # Minimal vendored OCP subset (BaseModel, LmdbDataset, Normalizer)
`-- train_scripts/                        # Convenience shell scripts
```

## Third-party code

`ocpmodels/` contains the small subset of the legacy Open Catalyst Project
codebase (`https://github.com/Open-Catalyst-Project/ocp`, MIT license, see
`ocpmodels/LICENSE`) that this repository imports: `BaseModel`, `LmdbDataset` and the target `Normalizer`, together with the registry,
typing and utility helpers they depend on. The original repository has since
moved to Fair-Chem, whose API no longer preserves these import paths, so the
files are kept here unchanged apart from trimming `datasets/__init__.py`.

## Installation

Create the conda environment:

```bash
conda env create -f ue.yml
conda activate ue
```

Run all commands from the repository root. The training scripts add the repository
root to `PYTHONPATH` internally, so an editable install is not required.

## Data

The code expects datasets under `dataset/`.

For MD17 and MD17-RotOOD, use the following layout:

```text
dataset/
|-- aspirin/
|   `-- raw/md17_aspirin.npz
|-- ethanol/
|   `-- raw/md17_ethanol.npz
|-- malonaldehyde/
|   `-- raw/md17_malonaldehyde.npz
|-- naphtalene/
|   `-- raw/md17_naphthalene.npz
|-- salicylicacid/
|   `-- raw/md17_salicylic.npz
|-- toluene/
|   `-- raw/md17_toluene.npz
`-- uracil/
    `-- raw/md17_uracil.npz
```

Processed files are generated automatically on first use. For RotOOD experiments,
`MD17Aligned` writes aligned samples to `processed_aligned_thr<threshold>/`, and the held-out
test split is wrapped by `RotatedDataset` to apply a 90-degree rotation.

For QM7X, place the raw HDF5 files under `dataset/QM7X/raw/`:

```text
dataset/QM7X/raw/1000.hdf5
dataset/QM7X/raw/2000.hdf5
...
dataset/QM7X/raw/8000.hdf5
```

## Running experiments

### Smoke test

Use `--test True` to restrict the run to a small subset and at most 10 epochs:

```bash
python parallel_lightning.py \
  --config configs/aspirin/config_painn_emer_aspirin.yml \
  --test True
```

### Single MD17-RotOOD EMER run

```bash
python parallel_lightning.py \
  --config configs/aspirin/config_painn_emer_aspirin.yml
```

### Single backbone run

```bash
python parallel_lightning.py \
  --config configs/aspirin/config_painn_backbone_aspirin.yml
```

### Run all included aspirin configs

```bash
for config in configs/aspirin/*.yml; do
  python parallel_lightning.py --config "$config"
done
```

### Submitit or Slurm execution

On clusters supported by `submitit`, add `--submitit True`:

```bash
python parallel_lightning.py \
  --config configs/aspirin/config_painn_emer_aspirin.yml \
  --submitit True
```

## Outputs

Each run writes results to:

```text
runs/<month-day>/<dataset>_<method>_<timestamp>/
```

Important files include:

- `results.txt`: per-seed metrics plus mean and standard deviation.
- `energy_pred_*.pt`, `energy_true_*.pt`: saved energy predictions and labels.
- `force_pred_*.pt`, `force_true_*.pt`: saved force predictions and labels.
- `force_stds_*.pt`: predicted force covariance or uncertainty tensor.
- TensorBoard and CSV logger outputs under the run directory.

The reported metrics include energy MAE, force MAE, Mahalanobis calibration
score, dimension-wise calibration, and dimension-wise NLL.

In run directory names, `ForceEvi_PaiNN` denotes the EMER model, and
`ForceEvi_PaiNNDropout22` denotes the dropout EMER variant.

## Method variants

The main config knobs are in `model_args` and `train_args`:

- `loss_fn: EnergyL1ForceEvidentialLoss` enables vector evidential force
  uncertainty.
- `use_alpha: true` adds the isotropic positive-definite term `exp(alpha) I`.
- `l_dim: 2` uses the rank-2 equivariant low-rank term in EMER.
- `dataset: <molecule>_ood` enables the MD17-RotOOD aligned train/test protocol.

The currently included EMER configs use PaiNN with `use_alpha: true` and
`l_dim: 2`.

## Tests

A small calibration unit test is available:

```bash
python -m pytest test/metrices/test_calibration_scores.py
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2026preserving,
  title     = {Preserving Geometric Symmetry in Uncertainty Estimation for Molecular Forces},
  author    = {Li, Xinyu and Zhang, Zhen and Huang, Jiayu and Steinberg, Daniel M. and Yao, Lina and Shi, Javen Qinfeng},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ey4IT8N8rl}
}
```
