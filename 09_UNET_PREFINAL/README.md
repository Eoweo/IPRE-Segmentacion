# 09_UNET_PREFINAL – Final Model with W&B Integration

## Objective

Final model training using PyTorch Lightning and full W&B integration for experiment tracking.

## Contents

- `train_unet_prefinal.py`
- `UNetLightning.py` and DataModule.
- W&B logging and model checkpointing.

## Setup

```bash
pip install pytorch-lightning wandb torch torchvision
```

Log in to Weights & Biases:

```bash
wandb login
```

## Usage

```bash
python train_unet_prefinal.py
```

## Output

- W&B dashboard with logs, images, and model checkpoints.
- Final model checkpoint for deployment or inference.
