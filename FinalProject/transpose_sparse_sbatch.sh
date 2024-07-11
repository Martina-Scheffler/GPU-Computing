#!/bin/bash
#SBATCH --job-name=transpose_sparse
#SBATCH --output=tps_%j.out
#SBATCH --error=tps_%j.err
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

# clean before run
rm -rf output/
mkdir output/

# run
srun ./bin/transpose 3 all 1