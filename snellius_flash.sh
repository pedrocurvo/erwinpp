#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=erwin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1
module load 2023
module load CUDA/12.4.0


source activate erwin
cd $HOME/erwinpp

srun python train_shapenet_flash.py --data-path shapenet_car/mlcfd_data/preprocessed --experiment shapenet_flash