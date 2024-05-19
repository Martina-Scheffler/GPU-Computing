#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>


#include "../include/matrix_generation.h"

using namespace std;

#define TILE_DIMENSION 4
#define BLOCK_ROWS 1

__global__ void transposeSimple(int* A, int* A_T){
	int x = blockIdx.x * TILE_DIMENSION + threadIdx.x;
	int y = blockIdx.y * TILE_DIMENSION + threadIdx.y;
    int width = gridDim.x * TILE_DIMENSION;

    for(int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        A_T[x * width + (y + i)] = A[(y + i) * width + x];
    }
}

__global__ void transposeCoalesced(int *A, int *A_T){
    __shared__ int tile[TILE_DIMENSION][TILE_DIMENSION + 1];  // +1 in y to avoid bank conflicts

    int x = blockIdx.x * TILE_DIMENSION + threadIdx.x;
    int y = blockIdx.y * TILE_DIMENSION + threadIdx.y;
    int width = gridDim.x * TILE_DIMENSION;

    for (int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        tile[threadIdx.y + i][threadIdx.x] = A[(y + i) * width + x];
    }
        
    __syncthreads();

    x = blockIdx.y * TILE_DIMENSION + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIMENSION + threadIdx.y;

    for (int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        A_T[(y + i) * width + x] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeDiagonal(int *A, int *A_T){
    __shared__ int tile[TILE_DIMENSION][TILE_DIMENSION + 1];

    // diagonal reordering
    int blockIdx_y = blockIdx.x;
    int blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    int x = blockIdx_x * TILE_DIMENSION + threadIdx.x;
    int y = blockIdx_y * TILE_DIMENSION + threadIdx.y;
    int width = gridDim.x * TILE_DIMENSION;


    for (int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        tile[threadIdx.y + i][threadIdx.x] = A[(y + i) * width + x];
    }
        
    __syncthreads();

    x = blockIdx_y * TILE_DIMENSION + threadIdx.x;  
    y = blockIdx_x * TILE_DIMENSION + threadIdx.y;

    for (int i=0; i<TILE_DIMENSION; i+=BLOCK_ROWS){
        A_T[(y + i) * width + x] = tile[threadIdx.x][threadIdx.y + i];
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

        cout << size << endl;

		// call matrix generation with command line argument and receive matrix back
		int* A = generate_continous_matrix(size);

        for (int i=0; i<size; i++){
            for (int j=0; j<size; i++){
                cout << i*size + j << "\t";
            }
            cout << "\n";
        }

        // allocate memory on host
        int* A_T = (int*) malloc(N * sizeof(int));

        // allocate memory on device
        int *dev_A, *dev_A_T;

        cudaMalloc(&dev_A, N * sizeof(int));
        cudaMalloc(&dev_A_T, N * sizeof(int));

        // copy matrix to device
        cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

        // start CUDA timer

        // determine kernel dimensions
        dim3 nBlocks (size / TILE_DIMENSION, size / TILE_DIMENSION, 1);
        dim3 nThreads (TILE_DIMENSION, BLOCK_ROWS, 1);

        // run kernel
        transposeSimple<<<nBlocks, nThreads>>>(dev_A, dev_A_T);
        // transposeCoalesced<<<nBlocks, nThreads>>>(A, A_T);
        // transposeDiagonal<<nBlocks, nThreads>>>(A, A_T)

        // synchronize
        cudaDeviceSynchronize();

        // stop CUDA timer

        // copy back
        cudaMemcpy(dev_A_T, A_T, N * sizeof(int), cudaMemcpyDeviceToHost);

        // display result
        // for (int i=0; i<size; i++){
        //     for (int j=0; j<size; i++){
        //         cout << A_T[i*size + j] << "\t";
        //     }
        //     cout << "\n";
        // }
        
        // free memory on device
        cudaFree(dev_A);
        cudaFree(dev_A_T);

        // free memory on host
        free(A);
        free(A_T);

        return 0;
    }

}