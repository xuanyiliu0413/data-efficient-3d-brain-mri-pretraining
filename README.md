# Data-Efficient Pretraining of 3D Foundation Models for Brain MRI

Research code for investigating how pretraining data scale and
self-supervised learning objectives affect downstream brain MRI performance.

## Overview

This project compares two approaches to self-supervised pretraining:

- **SelfMedMAE:** masked reconstruction of 3D MRI volumes.
- **3D JEPA:** latent representation prediction with an EMA target encoder.

Both approaches use 3D Vision Transformer encoders and are evaluated
at four nested pretraining data fractions: 12.5%, 25%, 50%, and 100%
of the FOMO50K training corpus.

## Downstream Evaluation

Three complementary tasks are investigated:

| Dataset | Task | Primary metric |
|---|---|---|
| IXI | Brain-age regression | Held-out test MAE (years) |
| ADNI | Alzheimer's disease vs cognitively normal classification | Cross-validation validation AUC |
| BraTS 2021 | Brain tumour segmentation | Best validation Dice |

The experiments assess whether the relative performance of
reconstruction-based and predictive pretraining changes with
data scale and downstream task.

## Repository Structure

- `implementation/selfmedmae/`: masked autoencoder implementation.
- `implementation/neurojepa/`: predictive representation learning implementation.
- `pretraining/jepa/`: JEPA pretraining entry point.
- `configs/pretrain/`: pretraining configuration files.
- `configs/downstream/`: downstream experiment configurations.
- `downstream/ixi/`: brain-age regression.
- `downstream/adni/`: AD/CN classification.
- `downstream/brats/`: tumour segmentation.
- `preprocessing/`: data preprocessing utilities.
- `results/`: non-identifying aggregate experimental results.
- `third_party_licenses/`: upstream software licences.
- `docs/THIRD_PARTY_NOTICES.md`: third-party attribution.

## Pretraining Configuration

Both methods use an upstream batch size of 16 and 100 training epochs.

For SelfMedMAE, the configured base learning rate is 6.4e-4,
scaled by batch_size / 256 to 4e-5, with a one-epoch warm-up.

For JEPA, the learning rate is 4e-5 and the archived configurations
specify fixed EMA momentum values of [0.99925, 0.99925].


## Data and Model Weights

This repository does not distribute MRI volumes, subject-level
predictions, private experiment logs, pretrained checkpoints,
or restricted third-party weights.

Users must obtain appropriate dataset permissions and configure
local data and checkpoint paths.

Relevant environment variables include:

- `BRAINMRI_ADNI_ROOT`
- `BRAINMRI_ADNI_DATA_ROOT`
- `BRAINMRI_IXI_DATA_ROOT`
- `BRAINMRI_PROJECT_ROOT`
- `BRAINMRI_JEPA_ENCODER_DIR`


## Acknowledgements and Licences

This repository includes adapted components from SelfMedMAE
and Neuro-JEPA.

Applicable upstream licences and attribution notices are
preserved in `third_party_licenses/` and
`docs/THIRD_PARTY_NOTICES.md`.

## Installation

Python 3.11 is recommended.

    git clone https://github.com/xuanyiliu0413/data-efficient-3d-brain-mri-pretraining.git
    cd data-efficient-3d-brain-mri-pretraining
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Configuration

Pretraining configurations are available under
`configs/pretrain/`. The four JEPA configurations refer to
separate training manifests under `data/fomo50k/manifests/`.

Dataset and checkpoint files are not distributed.
Local input paths must be configured for authorised datasets.

Manifest structure and dataset sizes are described in
`manifests/README.md`.

## Testing

    python -m unittest discover -s tests -v
    python scripts/check_manifests.py

The manifest check requires the corresponding local CSV files.

## Statistical analysis

`analysis/paired_bootstrap.py` provides paired resampling
for case-level metric comparisons.

Input CSV columns: `case_id`, `mae`, `jepa`.

The utility can be used with synthetic or authorised
case-level scores.

## Licence

Original project contributions are distributed under Apache-2.0.
Third-party components retain their respective upstream licences.
See `LICENSE`, `third_party_licenses/` and
`docs/THIRD_PARTY_NOTICES.md`.
