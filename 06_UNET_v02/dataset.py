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

def tif_dataset_generator(abs_path, mode="test"):
 
    training_file = os.path.join(abs_path, 'training.tif')
    training_mask_file = os.path.join(abs_path, 'training_groundtruth.tif')
    testing_file = os.path.join(abs_path, 'testing.tif')
    testing_mask_file = os.path.join(abs_path, 'testing_groundtruth.tif')

    def process_tif(file_path):
        try:
            with Image.open(file_path) as img:
                while True:
                    image = img.convert("L")
                    if p.RESIZE:
                        image = image.resize(p.RESIZE_VALUE)
                    
                    yield np.array(image) / 255.0
                    
                    img.seek(img.tell() + 1)
        except EOFError:
            return  
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    if mode == "train":
        img_gen = process_tif(training_file)
        mask_gen = process_tif(training_mask_file)
    elif mode == "test":
        img_gen = process_tif(testing_file)
        mask_gen = process_tif(testing_mask_file)

    for img, mask in zip(img_gen, mask_gen):
        yield img, mask



PATIENT_SPLITS = {"train": set(), "test": set()}

def initialize_patient_splits(abs_path, test_ratio=p.RATIO):
 
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


def load_jpg_dataset_generator(abs_path, dataset_type="test", target_size=(128, 128), block_id=set()):
     
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
    brightness = random.uniform(min_range, max_range)
    v = image
    if brightness >= 0:
        lim = (1 - brightness)
        v[v > lim] = 1
        v[v <= lim] += brightness
    else:
        brightness = abs(brightness)
        lim = brightness
        v[v < lim] = 0
        v[v >= lim] -= brightness
    return v

def random_contrast(image, min_range=0.1, max_range=2.5):
    contrast = random.uniform(min_range, max_range)
    mean = np.mean(image)
    #print(f"Formula: (image - {mean}) * {contrast} + {mean}")
    #print(f"Before Clipping: Min = {image.min()}, Max = {image.max()}")

    image = np.clip((image - mean) * contrast + mean, 0, 1)
    #print(f"After Clipping: Min = {image.min()}, Max = {image.max()}")

    return image

def random_rotate(image, mask, min_angle, max_angle):
    angle = random.randint(min_angle, max_angle)
    rotated_image = ndimage.rotate(image, angle, reshape=False)
    rotated_mask = ndimage.rotate(mask, angle, reshape=False)
    return rotated_image, rotated_mask

def apply_augmentations(image, mask, n=1, augmentation_prob=0.8):
    image = c.deepcopy(image)
    mask = c.deepcopy(mask)
    list = []

    for _ in range(n):
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_horizontal_flip(image, mask, flip_prob=0.5)
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_crop(image, mask, scale=0.9)
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_padding(image, mask, padding_range=max(int(round(p.RESIZE_VALUE[0]*0.025, 0)), 1))
        if random.uniform(0, 1.0) < augmentation_prob:
            image = random_brightness(image, -0.1, 0.3/p.N_AUGMENTATION) #-.5 - 0.5max
        if random.uniform(0, 1.0) < augmentation_prob:
           image = random_contrast(image, 0.7, 2/(p.N_AUGMENTATION)) #0.1 - 2.5max
        if random.uniform(0, 1.0) < augmentation_prob:
            image, mask = random_rotate(image, mask, -10, 10)
    return image, mask



# Define the Albumentations augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        # Geometric transformations (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=512, width=512, p=0.8),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=0.8),
        A.Rotate(limit=10, p=0.8),

        # Image-only transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=(0.7, 2.0), p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.8),

        # Resize back to ensure consistent output size
        A.Resize(p.RESIZE_VALUE[0], p.RESIZE_VALUE[1] ),

        # Convert to PyTorch tensors
        ToTensorV2()
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
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Convert to tensor if no augmentation
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  

        return image, mask

#def set_tif_dataset(abs_path):
#
#    training_file = os.path.join(abs_path, 'training.tif')
#    training_mask_file = os.path.join(abs_path, 'training_groundtruth.tif')
#    testing_file = os.path.join(abs_path, 'testing.tif')
#    testing_mask_file = os.path.join(abs_path, 'testing_groundtruth.tif')
#
#    def process_tif(file_path):
#        images_list = []
#        try:
#            with Image.open(file_path) as img:
#                while True:
#                    img_gray = img.convert("L")
#                    if p.RESIZE:
#                        img_resized = img_gray.resize(p.RESIZE_VALUE)
#
#                    images_list.append(np.array(img_resized) / 255.0)
#
#                    img.seek(img.tell() + 1)
#        except EOFError:
#            pass  
#        except FileNotFoundError:
#            raise FileNotFoundError(f"File not found: {file_path}")
#
#        return np.stack(images_list) if images_list else None
#
#    training = process_tif(training_file)
#    training_mask = process_tif(training_mask_file)
#    test = process_tif(testing_file)
#    test_mask = process_tif(testing_mask_file)
#
#    return test, test_mask, training, training_mask
class MainDataseta(Dataset):
    def __init__(self, data, augmentation, type = "test"):
        self.images = []
        self.masks = []
        self.type = type
        self.augmentation = augmentation

        # Process the original dataset from the generator
        for img, mask in data:
            self.images.append(img)
            self.masks.append(mask)

        # Convert to NumPy arrays
        self.images = self.images #np.array(self.images, dtype=np.float32)
        self.masks =  self.masks  #np.array(self.masks , dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = c.deepcopy(self.images[idx]), c.deepcopy(self.masks[idx])
        #print(f"beforeimage:{idx} - shape: {image.shape}  - max{np.max(image)} - min: {np.min(image)} ")

        if self.augmentation and self.type == "Training":
            list = []
            image, mask = apply_augmentations(image, mask, n=p.N_AUGMENTATION, augmentation_prob=0.8)
            #print(f"image:{idx} - shape: {image.shape} max{np.max(image)} - min: {np.min(image)} - {list}")
        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (H, W) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  

        return image, mask