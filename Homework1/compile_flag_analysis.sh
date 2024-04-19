#!/bin/bash

# Cleanup after previous usage
rm -rf bin/
rm -rf output/
mkdir output/

# make with set flag -O0
export USER_COMPILE_FLAGS=-O0
make

# run
./bin/simple_transpose 0 00
./bin/block_transpose 0 32 00

# repeat for flag -O1
rm -rf bin/

export USER_COMPILE_FLAGS=-O1
make

./bin/simple_transpose 0 01
./bin/block_transpose 0 32 01

# repeat for flag -O2
rm -rf bin/

export USER_COMPILE_FLAGS=-O2
make

./bin/simple_transpose 0 02
./bin/block_transpose 0 32 02

# repeat for flag -O3
rm -rf bin/

export USER_COMPILE_FLAGS=-O3
make

./bin/simple_transpose 0 03
./bin/block_transpose 0 32 03
