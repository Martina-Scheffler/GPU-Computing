#!/bin/bash

# Cleanup after previous usage
rm -rf bin/
rm -rf output/
mkdir output/
rm visualization/block*.png

# make with set flag -O0
export USER_COMPILE_FLAGS=-O0
make

# run
./bin/block_transpose 0 2 00_2
./bin/block_transpose 0 4 00_4 
./bin/block_transpose 0 8 00_8
./bin/block_transpose 0 16 00_16
./bin/block_transpose 0 32 00_32
./bin/block_transpose 0 64 00_64
./bin/block_transpose 0 128 00_128
./bin/block_transpose 0 256 00_256
./bin/block_transpose 0 512 00_512
./bin/block_transpose 0 1024 00_1024
./bin/block_transpose 0 2048 00_2048

# make with set flag -O1
rm -rf bin/
export USER_COMPILE_FLAGS=-O1
make

# run
./bin/block_transpose 0 2 01_2
./bin/block_transpose 0 4 01_4 
./bin/block_transpose 0 8 01_8
./bin/block_transpose 0 16 01_16
./bin/block_transpose 0 32 01_32
./bin/block_transpose 0 64 01_64
./bin/block_transpose 0 128 01_128
./bin/block_transpose 0 256 01_256
./bin/block_transpose 0 512 01_512
./bin/block_transpose 0 1024 01_1024
./bin/block_transpose 0 2048 01_2048

# make with set flag -O2
rm -rf bin/
export USER_COMPILE_FLAGS=-O2
make

# run
./bin/block_transpose 0 2 02_2
./bin/block_transpose 0 4 02_4 
./bin/block_transpose 0 8 02_8
./bin/block_transpose 0 16 02_16
./bin/block_transpose 0 32 02_32
./bin/block_transpose 0 64 02_64
./bin/block_transpose 0 128 02_128
./bin/block_transpose 0 256 02_256
./bin/block_transpose 0 512 02_512
./bin/block_transpose 0 1024 02_1024
./bin/block_transpose 0 2048 02_2048

# make with set flag -O3
rm -rf bin/
export USER_COMPILE_FLAGS=-O3
make

# run
./bin/block_transpose 0 2 03_2
./bin/block_transpose 0 4 03_4 
./bin/block_transpose 0 8 03_8
./bin/block_transpose 0 16 03_16
./bin/block_transpose 0 32 03_32
./bin/block_transpose 0 64 03_64
./bin/block_transpose 0 128 03_128
./bin/block_transpose 0 256 03_256
./bin/block_transpose 0 512 03_512
./bin/block_transpose 0 1024 03_1024
./bin/block_transpose 0 2048 03_2048