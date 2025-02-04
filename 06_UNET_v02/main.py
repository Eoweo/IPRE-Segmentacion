import os
import sys
import torch
import threading
import time
import GPUtil
import psutil
import shutil
import zipfile
import parameter as p
from model import UNet
from train import train_model
from visualization import Menu
from torch.utils.data import DataLoader

def system_monitor(interval=2):
    while True:
        used_memory =  psutil.virtual_memory().used / (1024**3)
        total_memory = psutil.virtual_memory().total / (1024**3)
        print(f"\nCPU: {psutil.cpu_percent()}% | RAM: {used_memory:.2f} / {total_memory:.2f} GB | " +
              (f"GPU: {GPUtil.getGPUs()[0].load * 100:.2f}%" ))
        time.sleep(interval)

if __name__ == "__main__":
    
    threading.Thread(target=system_monitor, args=(2,), daemon=True).start()


    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if p.SAVE_MODEL or p.SAVE_PLOTS:
        if os.path.exists(p.RESULT_DIR):
            shutil.rmtree(p.RESULT_DIR)  # Delete the entire directory
    if not os.path.exists(p.RESULT_DIR):        
        os.makedirs(p.RESULT_DIR)  # Recreate the directory
    Menu()


def get_next_zip_name(base_name="result_v0", folder=".", ext=".zip"):
    """Find the next available zip file name with incremental numbering."""
    version = 1
    while True:
        zip_name = f"{p.TEST_AVAILABLE[p.TEST_SELECTED_INDEX][:5].strip()}_{base_name}{version:02d}{ext}"
        if not os.path.exists(os.path.join(folder, zip_name)):
            return zip_name
        version += 1

def compress_folder(folder_to_zip):
    """Compress a folder into a uniquely named zip file."""
    # Ensure the folder exists
    if not os.path.exists(folder_to_zip):
        print(f"Error: Folder '{folder_to_zip}' does not exist.")
        return

    parent_directory = os.path.dirname(folder_to_zip)

    # Generate unique zip file names
    zip_name = get_next_zip_name(folder=parent_directory)
    print(parent_directory)
    print(zip_name)
    zip_path = os.path.join(parent_directory, zip_name)
    
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