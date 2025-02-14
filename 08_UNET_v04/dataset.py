import os
import gc
import cv2
import time
import torch
import psutil
import random
import copy as c
import numpy as np
import pandas as pd
import parameter as p
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import albumentations as A
from multiprocessing import Pool
from collections import defaultdict
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import rotate

def initialize_patient_splits(abs_path, test_ratio=p.RATIO):
 
    PATIENT_SPLITS = {"train": set(), "test": set()}
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    # Extract unique patient IDs
    patient_ids = list(set(row['ImageId'].split('_')[0] for _, row in data.iterrows()))

    # Shuffle and split
    random.shuffle(patient_ids)
    split_index = int(len(patient_ids) * test_ratio)
    
    PATIENT_SPLITS["test"] = set(patient_ids[split_index:])
    PATIENT_SPLITS["train"] = set(patient_ids[:split_index])

    print("Patient split initialized. Train:", len(PATIENT_SPLITS["train"]), "Test:", len(PATIENT_SPLITS["test"]))
    return PATIENT_SPLITS

def load_jpg_dataset_generator(abs_path, PATIENT_SPLITS, dataset_type="test", target_size=(128, 128), block_id=set()):
     
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    data = data[:p.CHOP_VALUE]

    image_dir = os.path.join(abs_path, "archive", "images", "images")
    mask_dir = os.path.join(abs_path, "archive", "masks", "masks")

    with tqdm(total=len(data), desc=f"Uploading {dataset_type} dataset", dynamic_ncols=True, leave=True) as pbar:
        i = 0
        for _, row in data.iterrows():
            image_name = row["ImageId"]
            mask_name = row["MaskId"]

            patient_id = image_name.split("_")[0]

            if patient_id in block_id or patient_id not in PATIENT_SPLITS[dataset_type]:
                pbar.update(1)
                continue
            i += 1

            image = Image.open(os.path.join(image_dir, image_name)).convert("L")
            mask = Image.open(os.path.join(mask_dir, mask_name)).convert("RGB")

            if p.RESIZE:
                image = np.array(image.resize(target_size), dtype=np.float32) / 255.0
                mask = np.array(mask.resize(target_size), dtype=np.float32) / 255.0
            else:
                image = np.array(image, dtype=np.float32) / 255.0
                mask = np.array(mask, dtype=np.float32) / 255.0

            threshold = 0.2
            mask = (mask[:, :, 2] > threshold) * mask[:, :, 2]

            pbar.update(1)
            used_memory =  psutil.virtual_memory().used / (1024**3)
            total_memory = psutil.virtual_memory().total / (1024**3)

            # Set progress bar postfix with estimated time left
            pbar.set_postfix({
                "Mem": f"{used_memory:.2f} / {total_memory:.2f} GB",
                "N_Img": f"{i}"})
            
            yield image, mask  # Instead of storing, yield one image at a time

def get_augmentation_pipeline():
    return A.Compose([

        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=512, width=512, p=0.8),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=0.8),
        A.Rotate(limit=10, p=0.8),

        # Image-only transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(0.7, 1.0), p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.8),

        A.Resize(p.RESIZE_VALUE[0], p.RESIZE_VALUE[1] ),
    ])

class MainDataset(Dataset):
    def __init__(self, data, augmentation=True, dataset_type="test"):
        self.images = []
        self.masks = []
        self.dataset_type = dataset_type
        self.augmentation = augmentation
        self.transform = get_augmentation_pipeline() if augmentation else None

        # Load dataset into memory
        for img, mask in data:
            self.images.append(img)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply augmentations if enabled
        if self.augmentation and self.dataset_type == "Training":
            augmented = self.transform(image=image, mask=mask)
            image = torch.tensor(augmented["image"], dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(augmented["mask"], dtype=torch.float32).unsqueeze(0)  
        else:
            # Convert to tensor if no augmentation
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  

        return image, mask