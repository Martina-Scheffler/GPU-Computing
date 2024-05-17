#!/bin/bash
#SBATCH --job-name=transpose
#SBATCH --output=tp_%j.out
#SBATCH --error=tp_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# load cuda
module load cuda

# build
make