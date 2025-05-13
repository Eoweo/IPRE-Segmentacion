# 03_Segmentation_NN – Basic Segmentation Network

## Objective

First step into segmentation: adapting a CNN for pixel-wise prediction using an encoder-decoder structure.

## Contents

- `Segmentation_NN.ipynb`: Implements a basic segmentation CNN.
- Loads image-mask pairs.
- Trains to predict segmentation masks.

## Setup

Ensure you have your dataset (images + masks). Install libraries:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

## Usage

```bash
jupyter notebook Segmentation_NN.ipynb
```

Update paths to your dataset and run the notebook.

## Output

- Sample predictions showing masks.
- Validation pixel accuracy or Dice score.
