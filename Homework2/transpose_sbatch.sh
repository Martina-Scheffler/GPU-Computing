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
module load cuda/12.1

# Clean before build
rm -rf bin/

# build
make

# clean after build
rm ./bin/*.o

# run
srun ./bin/transpose 2 0 4 1