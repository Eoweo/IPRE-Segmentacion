# 07_UNET_v03 – U-Net with Augmentation & Checkpoints

## Objective

Adds data augmentation, supports multi-class segmentation, and enables model checkpointing.

## Contents

- `train_unet_v03.py`
- Uses transforms for data augmentation.
- Saves best performing model.

## Setup

```bash
pip install torch torchvision albumentations numpy matplotlib
```

## Usage

```bash
python train_unet_v03.py
```

Make sure to set dataset and class count correctly.

## Output

- Augmented training improves generalization.
- Saved model and prediction samples.
