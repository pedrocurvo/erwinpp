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

# conda env create -f env.yaml


source activate erwin
cd $HOME/erwinpp

# srun python setup.py install

srun pip3 install torch==2.5.0 torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
srun pip3 install numpy einops Cython setuptools tqdm datasets transformers accelerate bitsandbytes auto-gptq scikit-learn huggingface_hub weasyprint matplotlib seaborn meshio open3d addict torch-cluster torch-scatter spconv-cu120 timm h5py tensorflow wandb matplotlib tqdm h5py tensorflow wandb matplotlib tqdm timm vtk pandas matplotlib seaborn meshio open3d addict spconv-cu120 timm h5py tensorflow wandb
srun pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4

srun python train_shapenet.py --data-path shapenet_car/mlcfd_data/preprocessed --experiment shapenet_default