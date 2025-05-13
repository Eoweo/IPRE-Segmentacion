# 05_UNET_v01 – U-Net Refactoring

## Objective

Refactors U-Net code into scripts and adds modularity for easier experimentation.

## Contents

- `train_unet_v01.py`: Training script.
- `model_unet.py`: U-Net architecture.
- Utility scripts.

## Setup

```bash
pip install torch torchvision numpy matplotlib
```

Ensure dataset path is correct in the script.

## Usage

```bash
python train_unet_v01.py
```

## Output

- Checkpoints saved during training.
- Console logs with training and validation losses.
