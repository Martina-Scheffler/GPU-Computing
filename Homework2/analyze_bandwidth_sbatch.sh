#!/bin/bash
#SBATCH --job-name=analyze_bandwidth
#SBATCH --output=bw_%j.out
#SBATCH --error=bw_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# load cuda
module load cuda

# Clean before build
rm -rf bin/

# build
make

# clean after build
rm ./bin/*.o

# run
srun ./bin/transpose 0 0 