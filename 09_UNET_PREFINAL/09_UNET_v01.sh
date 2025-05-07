#!/bin/bash
#SBATCH --job-name UNET_CT_Dataset # Custom name
#SBATCH -t 01-00:30:00 # Max runtime of 30 minutes
#SBATCH -p batch # Choose partition
#SBATCH -q batch # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 10 # Request 2 cores
#SBATCH --mem=100G # Indicate minimum memory
#SBATCH --gpus=2 # Do not use GPUs
#SBATCH -o /mnt/workspace/%u/slurm-out/%j.out # Write stdout to this file
#SBATCH -e /mnt/workspace/%u/slurm-out/%j.err # Write stderr to this file
#SBATCH --mail-type=ALL # Notify when it ends
#SBATCH --mail-user=mailto:camoro2002@uc.cl # Notify via email

## Here you can call your own script, for example:
module load conda
conda activate pyenv
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
srun torchrun --nproc-per-node=2 Segmentation.py
