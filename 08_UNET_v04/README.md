# 08_UNET_v04 – U-Net with PyTorch Lightning

## Objective

Refactors the training using PyTorch Lightning for modularity and multi-GPU support.

## Contents

- `train_unet_v04.py`
- `UNetLightning.py`: LightningModule for U-Net.
- Uses Trainer for training loop.

## Setup

```bash
pip install pytorch-lightning torch torchvision numpy matplotlib
```

## Usage

```bash
python train_unet_v04.py
```

## Output

- Lightning logs (lightning_logs/).
- Checkpoint files saved.
