import os
import sys
import torch
import time
import wandb
import GPUtil
import psutil
import shutil
import zipfile
import threading
import subprocess
import parameter as p
from model import UNet
from train import train_model
from visualization import Menu
from torch.utils.data import DataLoader


#def get_assigned_gpu():
#    """Detects which GPU the SLURM job is using."""
#    job_id = os.getenv("SLURM_JOB_ID")
#    if not job_id:
#        return None  # No SLURM job detected
#
#    try:
#        result = subprocess.run(
#            ["scontrol", "show", "job", job_id],
#            capture_output=True, text=True, check=True
#        )
#        for line in result.stdout.split("\n"):
#            if "GRES=gpu:" in line:
#                parts = line.split("GRES=gpu:")[-1]
#                gpu_ids = parts.split("(")[-1].split(")")[0]
#                return int(gpu_ids) if gpu_ids.isdigit() else None
#    except Exception:
#        return None
#
#def system_monitor(interval=2):
#    gpu_id = get_assigned_gpu()
#    
#    while True:
#        used_ram = psutil.virtual_memory().used / (1024**3)
#        total_ram = psutil.virtual_memory().total / (1024**3)
#
#        gpus = GPUtil.getGPUs()
#
#        gpu = gpus[gpu_id]
#        gpu_usage = f"GPU {gpus.uuid}: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB - {GPUtil.getGPUs()[0].load * 100:.2f}%"
#
#        print(f"CPU: {psutil.cpu_percent()}% | RAM: {used_ram:.2f}/{total_ram:.2f} GB | {gpu_usage}")
#        time.sleep(interval)

  # Change this to 1 if only monitoring one GPU

def get_gpu_memory():
    command = ["nvidia-smi", "--query-gpu=index,pci.bus_id,serial,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    lines = result.stdout.strip().split("\n")  # Limit to NUM_GPUS
    print(lines)
    print([line.split(", ") for line in lines])
    return [f"GPU {idx}: {bus_id} | Serial: {serial} | {used}/{total} MB | {util}%"
        for idx, bus_id, serial, used, total, util in [line.split(", ") for line in lines]]

def monitor_gpu(interval=2):
    """Print GPU memory usage for the selected number of GPUs."""
    while True:
        gpu_stats = get_gpu_memory()
        print(f"\nTime: {time.strftime('%H:%M:%S')}")
        print("\n".join(get_gpu_memory()))
        time.sleep(interval)

def system_monitor(interval=2):
    while True:
        used_memory =  psutil.virtual_memory().used / (1024**3)
        total_memory = psutil.virtual_memory().total / (1024**3)

        print(f"\nCPU: {psutil.cpu_percent()}% | RAM: {used_memory:.2f} / {total_memory:.2f} GB | " )
        print(f"gputil serial: {GPUtil.getGPUs()}") #(gpu.serial for gpu in GPUtil.getGPUs() )if len(GPUtil.getGPUs())>1 else 
        print("\n".join(get_gpu_memory()))
        time.sleep(interval)

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


if __name__ == "__main__":
    threading.Thread(target=system_monitor, args=(2,), daemon=True).start()
    #threading.Thread(target=monitor_gpu, args=(2,), daemon=True).start()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if p.SAVE_MODEL or p.SAVE_PLOTS:

        if os.path.exists(p.RESULT_DIR):
            shutil.rmtree(p.RESULT_DIR)  # Delete the entire directory
    if not os.path.exists(p.RESULT_DIR):        
        os.makedirs(p.RESULT_DIR)  # Recreate the directory
    wandb.init(project="unet-lung-segmentation", config={
    "epochs": p.EPOCHS,

    "batch_size": p.BATCH_SIZE,
    "learning_rate": p.LEARNING_RATE,
    "dataset": p.TEST_AVAILABLE[p.TEST_SELECTED_INDEX],
    "architecture": "UNet"
    })

    Menu()
    #compress_folder(p.RESULT_DIR)
    wandb.finish()