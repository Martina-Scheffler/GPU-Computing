#!/bin/bash
#SBATCH --job-name=compile_flag_analysis
#SBATCH --output=cfa_%j.out
#SBATCH --error=cfa_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

./compile_flag_analysis.sh