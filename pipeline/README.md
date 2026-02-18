# Offline pipeline

This component contains the offline processing + model training code for:
- **Self-collected headset data** (`pipeline.self.*`)
- **Public dataset training** (`pipeline.public.*`)

## Self-data quick baseline

```bash
eeg-self-train --data-dir data/self --runs 01 02 03 04 05 06 --simple-model rf
```

## Classical baselines (self data)

```bash
eeg-self-classical --subject 01 --data-dir data/self --runs 01 02 --sweep
```

## Public dataset

First create processed epochs locally:

```bash
python scripts/download_public_dataset.py --subjects 1 2 3 --runs 3 7 11
export PUBLIC_PROCESSED_DATA_DIR=data/public/processed_data
```

Then train:

```bash
eeg-public-train --pretrain
```

