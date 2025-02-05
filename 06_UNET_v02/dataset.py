import os
import gc
import cv2
import time
import torch
import psutil
import random
import numpy as np
import pandas as pd
import parameter as p
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from multiprocessing import Pool
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate

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

def initialize_patient_splits(abs_path, test_ratio=0.8):
 
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


def load_jpg_dataset_generator(abs_path, dataset_type="test", rotation=True, target_size=(128, 128), block_id=set()):
    """
    Generator function to load images and masks one by one, saving memory.
    """
    assert dataset_type in ["train", "test"], "dataset_type must be 'train' or 'test'"
    
    csv_path = os.path.join(abs_path, "archive", "train.csv")
    data = pd.read_csv(csv_path)

    data = data[:]

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

#https://github.com/kochlisGit/Data-Augmentation-Algorithms/blob/main/random_data_augmentations.py

def random_horizontal_flip(image, mask, flip_prob):
    if random.uniform(0, 1.0) < flip_prob:
        return image[:, ::-1].copy(), mask[:, ::-1].copy()
    return image, mask
def random_crop(image, mask, scale):
    height, width = image.shape[:2]
    x_min = int(width * scale)
    y_min = int(height * scale)
    x = random.randint(0, width - x_min)
    y = random.randint(0, height - y_min)
    
    cropped_image = image[y:y+y_min, x:x+x_min]
    cropped_mask = mask[y:y+y_min, x:x+x_min]
    return cv2.resize(cropped_image, (width, height)), cv2.resize(cropped_mask, (width, height))

def random_padding(image, mask, padding_range):
    padding_pixels = random.randint(0, padding_range)
    padded_image = cv2.copyMakeBorder(image, padding_pixels, padding_pixels, padding_pixels, padding_pixels, cv2.BORDER_CONSTANT)
    padded_mask = cv2.copyMakeBorder(mask, padding_pixels, padding_pixels, padding_pixels, padding_pixels, cv2.BORDER_CONSTANT)
    return cv2.resize(padded_image, image.shape[:2][::-1]), cv2.resize(padded_mask, mask.shape[:2][::-1])

def random_brightness(image, min_range, max_range):
    brightness = random.randint(min_range, max_range)
    v = image
    if brightness >= 0:
        lim = 255 - brightness
        v[v > lim] = 255
        v[v <= lim] += brightness
    else:
        brightness = abs(brightness)
        lim = brightness
        v[v < lim] = 0
        v[v >= lim] -= brightness
    return v

def random_contrast(image, min_range=0.5, max_range=1.5):
    contrast = random.uniform(min_range, max_range)
    mean = np.mean(image)
    image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
    return image

def random_rotate(image, mask, min_angle, max_angle):
    angle = random.randint(min_angle, max_angle)
    rotated_image = ndimage.rotate(image, angle, reshape=False)
    rotated_mask = ndimage.rotate(mask, angle, reshape=False)
    return rotated_image, rotated_mask

def apply_augmentations(image, mask, n=10, augmentation_prob=0.8):

    for _ in range(n):
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_horizontal_flip(image, mask, flip_prob=0.5)
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_crop(image, mask, scale=0.9)
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_padding(image, mask, padding_range=50)
        if random.uniform(0, 1.0) < augmentation_prob:
            image = random_brightness(image, -30, 30)
        if random.uniform(0, 1.0) < augmentation_prob:
            image = random_contrast(image, 0.5, 3)
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_rotate(image, mask, -70, 70)
        return image, mask

def augment_data(args):
    img, mask = args
    augmented_img, augmented_mask = apply_augmentations(img, mask)
    return augmented_img, augmented_mask

class MainDataset(Dataset):
    def __init__(self, data, num_workers = 8, type = "test"):
        self.images = []
        self.masks = []
        self.type = type

        # Process the original dataset from the generator
        for img, mask in data:
            self.images.append(img)
            self.masks.append(mask)

        # Convert to NumPy arrays
        self.images = np.array(self.images, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if p.AUGMENTATION and self.type != "Training":
            image, mask = apply_augmentations(image, mask, n=p.N_AUGMENTATION, augmentation_prob=0.8)


        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (H, W) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  

        return image, mask