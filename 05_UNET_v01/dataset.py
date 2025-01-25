import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import time

def set_tif_dataset(abs_path, target_size= (128, 128)):

    training_file = os.path.join(abs_path, 'training.tif')
    training_mask_file = os.path.join(abs_path, 'training_groundtruth.tif')
    testing_file = os.path.join(abs_path, 'testing.tif')
    testing_mask_file = os.path.join(abs_path, 'testing_groundtruth.tif')

    def process_tif(file_path):
        images_list = []
        try:
            with Image.open(file_path) as img:
                while True:
                    # Convert each page to a grayscale image
                    img_gray = img.convert("L")

                    # Resize to the desired dimensions
                    img_resized = img_gray.resize(target_size)

                    # Convert to a NumPy array and normalize pixel values to [0, 1]
                    img_array = np.array(img_resized) / 255.0

                    # Add to the list
                    images_list.append(img_array)

                    # Move to the next page
                    img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of the TIFF file
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        return np.stack(images_list) if images_list else None

    # Process each file
    training = process_tif(training_file)
    training_mask = process_tif(training_mask_file)
    test = process_tif(testing_file)
    test_mask = process_tif(testing_mask_file)

    return test, test_mask, training, training_mask

def set_jpg_Dataset(abs_path, rotation = True, target_size= (128, 128), block_id = [] ):
    csv_path = os.path.join(abs_path, 'train.csv')
    data = pd.read_csv(csv_path)
    data = data[:2000]

    image_dir = os.path.join(abs_path,'images/', 'images/')
    mask_dir =  os.path.join(abs_path,'masks/', 'masks/' ) 
    images_dict = defaultdict(list)
    masks_dict = defaultdict(list)

    start_time = time.time()
    i = 0

    with tqdm(total=len(data), desc="Img-mask upload", dynamic_ncols=True, leave=True) as pbar:
        for _, row in data.iterrows():
            image_name = row['ImageId']
            mask_name = row['MaskId']

            # Extract patient ID
            patient_id = image_name.split('_')[0]

            # Skip patient IDs in the block_id list
            if patient_id in block_id:
                pbar.update(1)
                continue

            angle = random.uniform(0, 180) if rotation else 0

            # Load and process the image
            image = Image.open(os.path.join(image_dir, image_name)).convert("L")
            image = image.rotate(angle)
            image = np.array(image.resize(target_size)) / 255.0

            mask = Image.open(os.path.join(mask_dir, mask_name)).convert("RGB")
            mask = mask.rotate(angle)
            mask = np.array(mask.resize(target_size)) / 255.0

            threshold = 0.2  # Adjust as needed
            mask = (mask[:, :, 2] > threshold) * mask[:, :, 2]

            images_dict[patient_id].append(image)
            masks_dict[patient_id].append(mask)

            # Update progress bar
            pbar.update(1)
            i += 1

            elapsed_time = time.time() - start_time
            remaining_steps = len(data) - pbar.n
            time_per_step = elapsed_time / pbar.n if pbar.n > 0 else 0
            estimated_time_left = time_per_step * remaining_steps

            # Set progress bar postfix with estimated time left
            pbar.set_postfix({
                "N_Img": f"{i}",
                "Estimated Time Left": f"{int(estimated_time_left // 60)}m {int(estimated_time_left % 60)}s"
            })

    # Randomly split patient IDs into training and testing sets
    patient_ids = list(images_dict.keys())
    #random.shuffle(patient_ids)
    
    split_index = len(patient_ids) // 2
    train_ids = patient_ids[:split_index]
    test_ids = patient_ids[split_index:]
    
    # Create training and testing datasets as NumPy arrays
    train_images = np.array([img for pid in train_ids for img in images_dict[pid]])
    train_masks = np.array([mask for pid in train_ids for mask in masks_dict[pid]])
    
    test_images = np.array([img for pid in test_ids for img in images_dict[pid]])
    test_masks = np.array([mask for pid in test_ids for mask in masks_dict[pid]])

    return train_images, train_masks, test_images, test_masks


class MainDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (H, W, C) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, mask