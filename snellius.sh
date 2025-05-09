#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=erwin_large_scale
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=outputs/erwin_output_%A.out

source erwin/bin/activate

cd experiments
python train_md.py --use-wandb 1 --size medium --model erwin --data-path "../data/shapenet_car/preprocessed"

# python main.py