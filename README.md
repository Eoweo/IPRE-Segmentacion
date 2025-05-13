# IPRE-Segmentacion – Deep Learning for Image Segmentation

This repository contains the step-by-step development of a deep learning pipeline for image segmentation using PyTorch. The project progresses from a simple neural network to a complete U-Net model, eventually incorporating advanced tools like **PyTorch Lightning** and **Weights & Biases (W&B)**.

---

## 🔍 Project Overview

The goal of this project is to segment images (such as rock samples or biomedical data) using deep learning. It begins with basic neural networks and ends with a scalable training system using PyTorch Lightning.

### Project Stages

Each folder corresponds to a stage of development:

| Folder | Description |
|--------|-------------|
| `01_Secuencial_NN` | Baseline with fully connected (sequential) neural network |
| `02_Convolutional_NN` | Convolutional Neural Network (CNN) for image classification |
| `03_Segmentation_NN` | First segmentation model (basic encoder-decoder) |
| `04_UNET_Segmentation` | Implementation of the U-Net architecture |
| `05_UNET_v01` | Modularized U-Net training with scripts |
| `06_UNET_v02` | Improvements in training, splitting, and structure |
| `07_UNET_v03` | Data augmentation and multi-class support |
| `08_UNET_v04` | PyTorch Lightning integration |
| `09_UNET_PREFINAL` | Final model using Lightning + W&B |

---

## ⚙️ Setup Instructions

To get started, create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install torch torchvision matplotlib numpy
pip install pytorch-lightning wandb albumentations jupyter
```

Make sure you log into W&B if using experiment tracking:

```bash
wandb login
```

---

## 🚀 Running the Models

Navigate to any folder and follow the instructions in its `README.md`.

Example:

```bash
cd 09_UNET_PREFINAL
python train_unet_prefinal.py
```

You can also start from earlier folders if you want to understand how the project evolved.

---

## 📊 Results

By the final stage, the model logs training metrics to **Weights & Biases**, including:

- Training and validation loss
- Dice score / IoU
- Sample predictions
- System usage and hyperparameters

---

## 📁 Dataset

This project assumes access to a dataset of images and their corresponding segmentation masks. You can adapt the dataset loading logic in each folder to match your data format.

---

## 🧠 Model

The final model is a U-Net architecture with:

- Convolutional encoder-decoder structure
- Skip connections
- Data augmentation
- PyTorch Lightning training
- W&B logging and early stopping

---

## 📌 Author Notes

This project was developed as part of a research and learning journey in biomedical imaging and computer vision. Each folder documents a meaningful step in the learning and implementation process.

Feel free to clone, modify, and use the code for educational purposes or research.

---

## 📜 License

MIT License (add if applicable)
