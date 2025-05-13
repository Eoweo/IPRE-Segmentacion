# 06_UNET_v02 – U-Net Improvements

## Objective

Fixes bugs, improves data splitting, and adjusts hyperparameters.

## Contents

- `train_unet_v02.py`: Enhanced training script.
- Dataset split into train/val/test.
- Configurable hyperparameters.

## Setup

Same as previous:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

```bash
python train_unet_v02.py
```

## Output

- Logs with training and validation metrics.
- Saved best model.
