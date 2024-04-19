#!/bin/bash
#SBATCH --job-name=block_size_analysis
#SBATCH --output=bsa_%j.out
#SBATCH --error=bsa_%j.err
#SBATCH --partition=edu5
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

srun ./block_size_analysis.sh