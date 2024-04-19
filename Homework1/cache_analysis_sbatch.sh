#!/bin/bash
#SBATCH --job-name=cache_analysis
#SBATCH --output=cache_%j.out
#SBATCH --error=cache_%j.err
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Clean before build
rm -rf bin/
rm -rf output/
mkdir output/
rm -rf valgrind/
mkdir valgrind/

# make with set flag -O3
export USER_COMPILE_FLAGS=-O3
make

# analyze using valgrind
srun valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/simple_transpose.out ./bin/simple_transpose 12
srun valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=valgrind/block_transpose.out ./bin/block_transpose 12