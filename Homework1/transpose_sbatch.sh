#!/bin/bash
#SBATCH --job-name=transpose
#SBATCH --output=tp_%j.out
#SBATCH --error=tp_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Clean before build
rm -rf bin/
rm -rf output/
mkdir output/

# choose compile flag: -O0, -O1, -O2, -O3
export USER_COMPILE_FLAGS=-O3
make

# run, choose algorithm: ./bin/simple_transpose, ./bin/block_transpose and matrix dimension 
./bin/block_transpose 12