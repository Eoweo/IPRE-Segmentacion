import os
import gc
import cv2
import json
import time
import torch
import psutil
import random as r
import copy as c
import numpy as np
import pandas as pd
import parameter as p
from PIL import Image
from tqdm import tqdm
import albumentations as A
from torch.utils.data import Dataset

def initialize_patient_splits(abs_path, test_ratio=p.RATIO):
    json_path = os.path.join("patient_splits.json")

    # Si el archivo ya existe, simplemente lo carga
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            PATIENT_SPLITS = json.load(f)
        # Convierte listas de nuevo a sets
        for k in PATIENT_SPLITS:
            PATIENT_SPLITS[k] = set(PATIENT_SPLITS[k])
        print(f"Patient split loaded from {json_path}")
        return PATIENT_SPLITS

    # Si no existe, lo genera y lo guarda
    PATIENT_SPLITS = {"train": set(), "validation": set(), "test": set()}
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    patient_ids = list(set(row['ImageId'].split('_')[0] for _, row in data.iterrows()if row['ImageId'].split('_')[0] not in p.BLOCK_ID))

    r.shuffle(patient_ids)

    if p.CHOP_PATIENT:
        split_index = int(round(p.CHOP_PATIENT_VALUE * test_ratio, 0))
        max_index = int(p.CHOP_PATIENT_VALUE)
    else:
        split_index = int(round((len(patient_ids)) * test_ratio, 0))
        max_index = int(len(patient_ids))
    
    val_index = int(round((max_index + split_index) / 2, 0))

    PATIENT_SPLITS["train"] = set(patient_ids[:split_index])
    PATIENT_SPLITS["validation"] = set(patient_ids[split_index:val_index])
    PATIENT_SPLITS["test"] = set(patient_ids[val_index:max_index])

    print("Patient split initialized. Train:", len(PATIENT_SPLITS["train"]),
          "Validation:", len(PATIENT_SPLITS["validation"]), "Test:", len(PATIENT_SPLITS["test"]))

    # Guarda en JSON (convierte los sets a listas)
    with open(json_path, "w") as f:
        json.dump({k: list(v) for k, v in PATIENT_SPLITS.items()}, f, indent=2)

    return PATIENT_SPLITS

def load_jpg_dataset_generator(abs_path, target_size=(128, 128), PATIENT_SPLITS = dict(), dataset_type="test", block_id=set()):

    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)
    if p.CHOP_DATA:
        data = data[:p.CHOP_DATA_VALUE]

    image_dir = os.path.join(abs_path, "archive", "images", "images")
    image_files = os.listdir(image_dir)
    mask_dir = os.path.join(abs_path, "archive", "masks", "masks")
    n_images = 0
    for image_file in image_files:
        patient_id = image_file.split('_')[0]
        if patient_id in PATIENT_SPLITS[dataset_type]:
                    n_images += 1        

    with tqdm(total=int(n_images), desc=f"Uploading {dataset_type} dataset", dynamic_ncols=True, ncols=80, ascii=False, leave=True) as pbar:
        i = 0
        for _, row in data.iterrows():
            image_id = row["ImageId"]
            mask_name = row["MaskId"]
            
            patient_id = image_id.split("_")[0]

            if patient_id in block_id or (patient_id not in block_id and patient_id not in PATIENT_SPLITS[dataset_type]): #pass if it's a block patient
                pbar.update(1)
                continue
            i += 1

            image = Image.open(os.path.join(image_dir, image_id)).convert("L")
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
            
            yield image, mask, image_id  # Instead of storing, yield one image at a time

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')

    img_eq = cdf_final[image]
    return img_eq.astype(np.float32) / 255.0

def get_adaptive_augmentation_pipeline(image):
    height, width = image.shape[:2]
    random_crop = r.randint(int(height/2), height)
    # Geometric transformations that affect both image and mask
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomSizedCrop(min_max_height=(random_crop - 1, random_crop), size=(random_crop , random_crop), p=0.8),
        A.PadIfNeeded(min_height=height, min_width=width, fill= 0, fill_mask= 0, p=1.0),
        A.Rotate(limit=25, p=0.8),
        A.Resize(height, width)
    ]

    return A.Compose(transforms, additional_targets={'mask': 'mask'})

class MainDataset(Dataset):
    def __init__(self, data, augmentation=p.AUGMENTATION, dataset_type="test"):
        self.images = []
        self.masks = []
        self.id = []
        self.dataset_type = dataset_type.lower() # prevenir error de mayus
        self.augmentation = augmentation

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

            
        if self.augmentation and self.dataset_type in ["train", "test"]:

            if r.randint(0,100) < 40:
                image = histogram_equalization(image)
            
            #convert for albumination to work
            image = (image * 255).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)
            transform = get_adaptive_augmentation_pipeline(image)  # <-- Call here
            augmented = transform(image=image, mask=mask)
            
            #we normalice again
            image = augmented["image"].astype(np.float32) / 255.0
            mask = augmented["mask"].astype(np.float32) / 255.0
        
        #make the tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask, id