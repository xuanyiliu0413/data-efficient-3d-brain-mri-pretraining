# Pretraining data

The training sets contain 6,084, 12,168, 24,336 and 48,672
volumes. A fixed validation set contains 491 volumes.

JEPA manifest files use a CSV column named `img_path`.

The corresponding local filenames are:

- `jepa_12p5_train.csv`
- `jepa_25_train.csv`
- `jepa_50_train.csv`
- `jepa_100_train.csv`

Place authorised local manifests under
`data/fomo50k/manifests/`.

The data directory is excluded from version control.
