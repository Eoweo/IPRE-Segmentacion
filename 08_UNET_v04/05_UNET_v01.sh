#!/bin/bash
#SBATCH --job-name UNET_CT_Dataset # Custom name
#SBATCH -t 3-00:00 # Max runtime of 30 minutes
#SBATCH -p batch # Choose partition
#SBATCH -q batch # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 2 # Request 2 cores
#SBATCH --mem=100G # Indicate minimum memory
#SBATCH --gpus=1 # Do not use GPUs
#SBATCH -o /mnt/workspace/%u/slurm-out/%j.out # Write stdout to this file
#SBATCH -e /mnt/workspace/%u/slurm-out/%j.err # Write stderr to this file
#SBATCH --mail-type=ALL # Notify when it ends
#SBATCH --mail-user=mailto:camoro2002@uc.cl # Notify via email

## Here you can call your own script, for example:
module load conda
conda activate pyenv
python Segmentation.py

## or run a notebook:
# jupyter notebook --port 30750 --no-browser
