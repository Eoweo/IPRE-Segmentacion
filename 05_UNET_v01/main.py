import os
import sys
import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import set_tif_dataset, set_jpg_Dataset, MainDataset
from train import train_model
import parameter as p
from visualization import Menu

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.makedirs(p.RESULT_DIR, exist_ok=True)
    Menu()
