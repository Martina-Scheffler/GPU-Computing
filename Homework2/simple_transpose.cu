#include <stdio.h>
#include <stdlib.h>

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


int main(void){
    int* A, A_T;
    N = 4;
	dim3 nBlocks = (N / TILE_DIMENSION, N / TILE_DIMENSION, 1);
    dim3 nThreads = (TILE_DIMENSION, BLOCK_ROWS, 1);
	simpleTransposeKernel<<<nBlocks, nThreads>>>(A, A_T);
	cudaDeviceSynchronize();
	return 0;
}