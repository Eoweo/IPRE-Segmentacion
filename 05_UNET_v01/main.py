import os
import sys
import torch
from torch.utils.data import DataLoader
import zipfile
from model import UNet
from dataset import set_tif_dataset, set_jpg_Dataset, MainDataset
from train import train_model
import parameter as p
from visualization import Menu

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.makedirs(p.RESULT_DIR, exist_ok=True)
    Menu()

def get_next_zip_name(base_name="result_v0", folder=".", ext=".zip"):
    """Find the next available zip file name with incremental numbering."""
    version = 1
    while True:
        zip_name = f"{base_name}{version:02d}{ext}"
        if not os.path.exists(os.path.join(folder, zip_name)):
            return zip_name
        version += 1

def compress_folder(folder_to_zip, output_folder="."):
    """Compress a folder into a uniquely named zip file."""
    # Ensure the folder exists
    if not os.path.exists(folder_to_zip):
        print(f"Error: Folder '{folder_to_zip}' does not exist.")
        return

    parent_directory = os.path.dirname(folder_to_zip)

    # Generate unique zip file name
    zip_name = get_next_zip_name(folder=parent_directory)
    zip_path = os.path.join(output_folder, zip_name)
    
    # Compress the folder
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_to_zip)
                zipf.write(file_path, arcname)
    
    print(f"Folder '{folder_to_zip}' has been compressed into '{zip_path}'.")

# Example usage
compress_folder(p.RESULT_DIR)