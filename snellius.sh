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

# srun pip3 install torch==2.5.0 
# srun pip3 install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
# srun pip3 install numpy 
# srun pip3 install einops
# srun pip3 install Cython
# srun pip3 install setuptools
# srun pip3 install tqdm
# srun pip3 install datasets
# srun pip3 install transformers
# srun pip3 install accelerate
# srun pip3 install bitsandbytes
# srun pip3 install auto-gptq
# srun pip3 install scikit-learn
# srun pip3 install huggingface_hub
# srun pip3 install weasyprint
# srun pip3 install python-kaleido
# srun pip3 install torch
# srun pip3 install numpy
# srun pip3 install pandas
# srun pip3 install matplotlib
# srun pip3 install seaborn
# srun pip3 install meshio open3d
# srun pip install addict
# srun pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html # CUDA 12.4
# srun pip install spconv-cu120
# srun pip install timm
# srun pip install h5py
# srun pip install tensorflow
# srun pip install wandb
# srun pip install matplotlib
# srun pip install tqdm
# 

srun python train_shapenet.py --data-path shapenet_car/mlcfd_data/preprocessed --experiment shapenet_default