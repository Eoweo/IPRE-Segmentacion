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
    if p.CHOP_PATIENT:
        split_index = int(round(p.CHOP_PATIENT_VALUE * test_ratio, 0))
        max_index = int(p.CHOP_PATIENT_VALUE)
    else:
        split_index = int(round((len(patient_ids)) * test_ratio, 0))
        max_index = int(len(patient_ids))

    max_index = int(round(split_index/p.RATIO, 0))
    PATIENT_SPLITS["train"] = set(patient_ids[:split_index])
    PATIENT_SPLITS["test"] = set(patient_ids[split_index:max_index])

    print("Patient split initialized. Train:", len(PATIENT_SPLITS["train"]), "Test:", len(PATIENT_SPLITS["test"]))
    return PATIENT_SPLITS

def load_jpg_dataset_generator(abs_path, target_size=(128, 128), PATIENT_SPLITS = dict(), dataset_type="test", block_id=set(), inference = p.INFERENCE):

    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)
    if p.CHOP_DATA:
        data = data[:p.CHOP_DATA_VALUE]

    image_dir = os.path.join(abs_path, "archive", "images", "images")
    mask_dir = os.path.join(abs_path, "archive", "masks", "masks")

    with tqdm(total=len(data), desc=f"Uploading {dataset_type} dataset", dynamic_ncols=True, leave=True) as pbar:
        i = 0
        for _, row in data.iterrows():
            image_name = row["ImageId"]
            mask_name = row["MaskId"]

            patient_id = image_name.split("_")[0]

            if not inference: 
                if patient_id in block_id or (patient_id not in block_id and patient_id not in PATIENT_SPLITS[dataset_type]): #pass if it's a block patient
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
            mask = (mask[:, :, 2] > threshold) * mask[:, :, 2] # takes the blue color of the mask only

            pbar.update(1)
            used_memory =  psutil.virtual_memory().used / (1024**3)
            total_memory = psutil.virtual_memory().total / (1024**3)

            # Set progress bar postfix with estimated time left
            pbar.set_postfix({
                "Mem": f"{used_memory:.2f} / {total_memory:.2f} GB",
                "N_Img": f"{i}"})
            
            yield image, mask, image_name  # Instead of storing, yield one image at a time


def calculate_brightness_and_saturation(image):
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        # For grayscale: Brightness = mean pixel value, Saturation = 0
        brightness = np.mean(image) / 255.0
        saturation = 0.0
    else:  # RGB image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = hsv[..., 2].mean() / 255.0  # Value channel
        saturation = hsv[..., 1].mean() / 255.0   # Saturation channel
    return brightness, saturation

def get_adaptive_augmentation_pipeline(image):

    height, width = image.shape[:2]

    brightness, saturation = calculate_brightness_and_saturation(image)

    brightness_strength = max(0.1, 0.3 - brightness)
    contrast_strength = max(0.1, 1.0 - saturation)

    # Transformations that affect both image and mask
    geometric_transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=height, width=width, p=0.8),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=0.8),
        A.Rotate(limit=10, p=0.8),
    ]

    # Only apply color transforms to the image
    #color_transforms = [
    #    A.OneOf([
    #        A.RandomBrightnessContrast(
    #            brightness_limit=(-0.1, brightness_strength),
    #            contrast_limit=(0.5, contrast_strength),
    #            p=1.0
    #        ),
    #        A.ColorJitter(
    #            brightness=0.2 * (1 - brightness),
    #            contrast=0.2 * (1 - saturation),
    #            saturation=0.2 * (1 - saturation),
    #            hue=0.1,
    #            p=1.0
    #        ),
    #    ], p=0.8)
    #]

    # Full pipeline
    return A.Compose(
        transforms = geometric_transforms + #color_transforms + 
        [A.Resize(height, width)], additional_targets={'mask': 'mask'}
    )


def get_augmentation_pipeline():
    return A.Compose([

        A.HorizontalFlip(p=0.5),

        A.RandomCrop(p.RESIZE_VALUE[0],p.RESIZE_VALUE[1], p=0.8) if p.RESIZE 
                else A.RandomCrop(height=512, width=512, p=0.8), 
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=0.8),
        A.Rotate(limit=10, p=0.8),

        # Image-only transformations
        #A.OneOf([
        #    A.RandomBrightnessContrast(brightness_limit=(0.29, 0.3), contrast_limit=(0.5, .510), p=1.0),
        #    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        #], p=0),

        A.Resize(p.RESIZE_VALUE[0], p.RESIZE_VALUE[1] ),
    ])


def Set_tif_Dataset(path, resize):

    with Image.open(path) as img:
        try:
            while True:
                # Convert each page to an RGB image
                img_rgb = img.convert("L")

                # Resize to the desired dimensions
                img_resized = img_rgb.resize(resize)

                # Convert to a NumPy array and normalize pixel values to [0, 255]
                img_array = np.array(img_resized)/255.0

                # Add to the list
                yield img_array

                # Move to the next page
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of the TIFF file

def b_load_jpg_dataset_generator(abs_path, PATIENT_SPLITS, dataset_type="test", target_size=p.RESIZE_VALUE, block_id=set()):
     
    if dataset_type == "test": 
        image_path = os.path.join(p.PATH_DATASET,'EPFL', 'testing.tif')
        mask_path = os.path.join(p.PATH_DATASET, 'EPFL','testing_groundtruth.tif')
    else: 
        image_path = os.path.join(p.PATH_DATASET,'EPFL', 'training.tif')
        mask_path = os.path.join(p.PATH_DATASET, 'EPFL','training_groundtruth.tif')
    
    image_gen = Set_tif_Dataset(image_path, target_size)
    mask_gen = Set_tif_Dataset(mask_path, target_size)

    total_samples = sum(1 for _ in Set_tif_Dataset(image_path, target_size))

    with tqdm(total=total_samples, desc=f"Uploading {dataset_type} dataset", dynamic_ncols=True, leave=True) as pbar:
        i = 0
        for img_array, mask_array in zip(image_gen, mask_gen):
            pbar.update(1)

            used_memory = psutil.virtual_memory().used / (1024**3)
            total_memory = psutil.virtual_memory().total / (1024**3)
            pbar.set_postfix({
                "Mem": f"{used_memory:.2f} / {total_memory:.2f} GB",
                "N_Img": f"{i}"})
            
            i += 1

            yield img_array, mask_array

class MainDataset(Dataset):
    def __init__(self, data, augmentation=p.AUGMENTATION, dataset_type="test"):
        self.images = []
        self.masks = []
        self.id = []
        self.dataset_type = dataset_type
        self.augmentation = augmentation
        self.transform = get_adaptive_augmentation_pipeline() if augmentation else None

        # Load dataset into memory
        for img, mask, id in data:
            self.images.append(img)
            self.masks.append(mask)
            self.id.append(id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        id = self.id[idx]

        # Normalize image values to [0, 255] if needed
        image = image.astype(np.uint8) if image.max() <= 1.0 else image

        if self.augmentation and self.dataset_type == "Training":
            transform = get_adaptive_augmentation_pipeline(image)  # <-- Call here
            augmented = transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert image and mask to torch tensors
        if len(image.shape) == 2:  # Grayscale
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:  # RGB
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask, id