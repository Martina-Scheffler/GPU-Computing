#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>

#include "include/helper_cuda.h"

#include "../include/matrix_generation.h"

using namespace std;

int TILE_DIMENSION = 4;
int BLOCK_ROWS = 1;

__global__ void simpleTransposeKernel(int* A, int* A_T){
	int x = blockIdx.x * TILE_DIMENSION + threadIdx.x;
	int y = blockIdx.y * TILE_DIMENSION + threadIdx.y;
    int width = gridDim.x * TILE_DIMENSION;

    for(int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        A_T[x * width + (y + i)] = A[(y + i) * width + x];
    }
}


int main(int argc, char* argv[]){
    // check if the matrix size was provided
	if (argc < 2){
		throw runtime_error("Please enter an integer N as argument to generate a matrix of size 2^N x 2^N.");
	}
    
    if (atoi(argv[1]) == 0){
        // use zero for something later
    }
    else {
        int size = pow(2, atoi(argv[1]));
        int N = size * size;

		// call matrix generation with command line argument and receive matrix back
		int* A = generate_continous_matrix(size);

        // allocate memory on device
        int *dev_A, *dev_A_t;

        checkCudaErrors( cudaMalloc(&dev_A, N * sizeof(float)) );
        checkCudaErrors( cudaMalloc(&dev_A_t, N * sizeof(float)) );

        // copy matrix to device
        cudaMemcpy(dev_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

        // start CUDA timer

        // determine kernel dimensions
        dim3 nBlocks (size / TILE_DIMENSION, size / TILE_DIMENSION, 1);
        dim3 nThreads (TILE_DIMENSION, BLOCK_ROWS, 1);

        // run kernel
        simpleTransposeKernel<<<nBlocks, nThreads>>>(dev_A, dev_A_t);

        // synchronize
        checkCudaErrors( cudaDeviceSynchronize() );

        // stop CUDA timer

        // allocate memory on host
        int* A_t = (int*) malloc(N * sizeof(float));

        // copy back
        cudaMemcpy(dev_A_t, A_t, N * sizeof(float), cudaMemcpyDeviceToHost);

        // display result
        
        // free memory on device
        checkCudaErrors( cudaFree(dev_A) );
        checkCudaErrors( cudaFree(dev_A_t) );

        // free memory on host
        free(A);
        free(A_t);

        return 0;
    }

}