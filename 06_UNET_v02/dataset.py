import os
import gc
import psutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import random
import time
import parameter as p

def set_tif_dataset(abs_path):

    training_file = os.path.join(abs_path, 'training.tif')
    training_mask_file = os.path.join(abs_path, 'training_groundtruth.tif')
    testing_file = os.path.join(abs_path, 'testing.tif')
    testing_mask_file = os.path.join(abs_path, 'testing_groundtruth.tif')

    def process_tif(file_path):
        images_list = []
        try:
            with Image.open(file_path) as img:
                while True:
                    img_gray = img.convert("L")
                    if p.RESIZE:
                        img_resized = img_gray.resize(p.RESIZE_VALUE)

                    images_list.append(np.array(img_resized) / 255.0)

                    img.seek(img.tell() + 1)
        except EOFError:
            pass  
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        return np.stack(images_list) if images_list else None

    training = process_tif(training_file)
    training_mask = process_tif(training_mask_file)
    test = process_tif(testing_file)
    test_mask = process_tif(testing_mask_file)

    return test, test_mask, training, training_mask

PATIENT_SPLITS = {"train": set(), "test": set()}

def initialize_patient_splits(abs_path, test_ratio=0.5):
 
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    # Extract unique patient IDs
    patient_ids = list(set(row['ImageId'].split('_')[0] for _, row in data.iterrows()))

    # Shuffle and split
    random.shuffle(patient_ids)
    split_index = int(len(patient_ids) * test_ratio)
    
    PATIENT_SPLITS["test"] = set(patient_ids[:split_index])
    PATIENT_SPLITS["train"] = set(patient_ids[split_index:])

    print("Patient split initialized. Train:", len(PATIENT_SPLITS["train"]), "Test:", len(PATIENT_SPLITS["test"]))


def load_jpg_dataset_generator(abs_path, dataset_type="test", rotation=True, target_size=(128, 128), block_id=set()):
    """
    Generator function to load images and masks one by one, saving memory.
    """
    assert dataset_type in ["train", "test"], "dataset_type must be 'train' or 'test'"
    
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    image_dir = os.path.join(abs_path, "archive", "images", "images")
    mask_dir = os.path.join(abs_path, "archive", "masks", "masks")

    with tqdm(total=len(data), desc=f"Uploading {dataset_type} dataset", dynamic_ncols=True, leave=True) as pbar:
        for _, row in data.iterrows():
            image_name = row["ImageId"]
            mask_name = row["MaskId"]

            patient_id = image_name.split("_")[0]

            if patient_id in block_id or patient_id not in PATIENT_SPLITS[dataset_type]:
                pbar.update(1)
                continue

            image = Image.open(os.path.join(image_dir, image_name)).convert("L")
            mask = Image.open(os.path.join(mask_dir, mask_name)).convert("RGB")

            if target_size:
                image = np.array(image.resize(target_size), dtype=np.float32) / 255.0
                mask = np.array(mask.resize(target_size), dtype=np.float32) / 255.0
            else:
                image = np.array(image, dtype=np.float32) / 255.0
                mask = np.array(mask, dtype=np.float32) / 255.0

            threshold = 0.2
            mask = (mask[:, :, 2] > threshold) * mask[:, :, 2]

            pbar.update(1)
            
            yield image, mask  # Instead of storing, yield one image at a time

class MainDataset(Dataset):
    def __init__(self, data, is_rotation):
        """
        Initialize the dataset with images and masks.
        :param data: A generator yielding (image, mask) pairs.
        :param is_rotation: If True, generates rotated images to augment the dataset.
        """
        self.images = []
        self.masks = []

        # Process the original dataset from the generator
        for img, mask in data:
            self.images.append(img)
            self.masks.append(mask)

            if is_rotation:
                angle = random.uniform(0, 270)  # Random rotation angle

                img_rotated = rotate(
                    torch.tensor(img, dtype=torch.float32).unsqueeze(0),
                    angle,
                    interpolation=Image.BILINEAR,
                    fill=0
                ).squeeze(0).numpy()

                mask_rotated = rotate(
                    torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
                    angle,
                    interpolation=Image.BILINEAR
                ).squeeze(0).numpy()

                self.images.append(img_rotated)
                self.masks.append(mask_rotated)

        # Convert to NumPy arrays
        self.images = np.array(self.images, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (H, W) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  

        return image, mask



#class MainDataset(Dataset):
#    def __init__(self, data, is_rotation):
#        self.data  = data
#        
#        if is_rotation:
#            rotated_images = []
#            rotated_masks = []
#
#            for img, mask in data:
#                angle = random.uniform(0, 270)
#
#                img_rotated = rotate(torch.tensor(img, dtype=torch.float32).unsqueeze(0), float(angle), interpolation=Image.BILINEAR, fill= 0)
#                mask_rotated = rotate(torch.tensor(mask, dtype=torch.float32).unsqueeze(0), angle, interpolation=Image.BILINEAR)
#
#                rotated_images.append(img_rotated.squeeze(0).numpy())
#                rotated_masks.append(mask_rotated.squeeze(0).numpy())
#
#            self.images = np.concatenate((self.images, np.array(rotated_images)), axis=0)
#            self.masks = np.concatenate((self.masks, np.array(rotated_masks)), axis=0)
#
#    def __len__(self):
#        return len(self.images)
#
#    def __getitem__(self, idx):
#        image = self.images[idx]
#        mask = self.masks[idx]
#
#        # Convert to PyTorch tensors
#        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (H, W, C) -> (C, H, W)
#        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#
#        yield image, mask