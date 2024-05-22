#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>


#include "../include/matrix_generation.h"

using namespace std;

#define NUM_REPS 10

int strategy = 0;
int tileDimension = 4;
int blockRows = 1;


__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
}

__global__ void transposeSimple(int* A, int* A_T, int tileDimension, int blockRows){
	int x = blockIdx.x * tileDimension + threadIdx.x;
	int y = blockIdx.y * tileDimension + threadIdx.y;
    int width = gridDim.x * tileDimension;

    for(int i=0; i<tileDimension; i+=blockRows){
        A_T[x * width + (y + i)] = A[(y + i) * width + x];
    }
}

__global__ void transposeCoalesced(int *A, int *A_T, int tileDimension, int blockRows){
    extern __shared__ int tile[];

    //__shared__ int tile[tileDimension][tileDimension + 1];  // +1 in y to avoid bank conflicts

    int x = blockIdx.x * tileDimension + threadIdx.x;
    int y = blockIdx.y * tileDimension + threadIdx.y;
    int width = gridDim.x * tileDimension;

    for (int i=0; i<tileDimension; i+=blockRows){
        tile[(threadIdx.y + i) * (tileDimension + 1) + threadIdx.x] = A[(y + i) * width + x];
    }
        
    __syncthreads();

    x = blockIdx.y * tileDimension + threadIdx.x;  // transpose block offset
    y = blockIdx.x * tileDimension + threadIdx.y;

    for (int i=0; i<tileDimension; i+=blockRows){
        A_T[(y + i) * width + x] = tile[threadIdx.x * (tileDimension + 1) + (threadIdx.y + i)];
    }
}

__global__ void transposeDiagonal(int *A, int *A_T, int tileDimension, int blockRows){
    extern __shared__ int tile[];
    //__shared__ int tile[tileDimension][tileDimension + 1];

    // diagonal reordering
    int blockIdx_y = blockIdx.x;
    int blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    int x = blockIdx_x * tileDimension + threadIdx.x;
    int y = blockIdx_y * tileDimension + threadIdx.y;
    int width = gridDim.x * tileDimension;


    for (int i=0; i<tileDimension; i+=blockRows){
        tile[(threadIdx.y + i) * (tileDimension + 1) + threadIdx.x] = A[(y + i) * width + x];
    }
        
    __syncthreads();

    x = blockIdx_y * tileDimension + threadIdx.x;  
    y = blockIdx_x * tileDimension + threadIdx.y;

    for (int i=0; i<tileDimension; i+=blockRows){
        A_T[(y + i) * width + x] = tile[threadIdx.x * (tileDimension + 1) + threadIdx.y + i];
    }
}


int main(int argc, char* argv[]){
    // check if the matrix size was provided
	if (argc < 2){
		throw runtime_error("Please enter an integer N as argument to generate a matrix of size 2^N x 2^N.");
	}
    if (argc >= 3){
        strategy = atoi(argv[2]);  // Strategy: 0 = Simple, 1 = Coalesced, 2 = Diagonal
        printf("Strategy %d\n", strategy);
    }
    if (argc >= 5){
        tileDimension = atoi(argv[3]);
        blockRows = atoi(argv[4]);
    }
    
    if (atoi(argv[1]) == 0){
        // use zero for analyzing the effective bandwidth with different matrix sizes and parameters

        // open file to store execution times
		std::ofstream myfile;
		string extension = argv[2]; // append extension to save output to the correct file
		myfile.open("output/analyze_bandwidth_" + extension + ".csv");

        for (int i=2; i<=pow(2, 12); i*=2){  // from 2^1 to 2^12
            // matrix dimension
            int size = i; 
            int N = size * size;

            // loop over all possible values of tile dimension and block rows
            for (int j=2; j<=i; j*=2){  // 2 to i/matrix dimension
                tileDimension = j;

                for (int k=1; k<=j; k*=2){  // 1 to j/tile dimension
                    blockRows = k;

                    // generate matrix
                    int* A = generate_random_matrix(size);

                    // allocate memory on host
                    int* A_T = (int*) malloc(N * sizeof(int));

                    // determine kernel dimensions
                    dim3 nBlocks (size / tileDimension, size / tileDimension, 1);
                    dim3 nThreads (tileDimension, blockRows, 1);

                    // allocate memory on device
                    int *dev_A, *dev_A_T;

                    cudaMalloc(&dev_A, N * sizeof(int));
                    cudaMalloc(&dev_A_T, N * sizeof(int));

                    // copy matrix to device
                    cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

                    // Create CUDA events to use for timing
                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);

                    // warmup to avoid timing startup TODO: is this necessary?
                    warm_up_gpu<<<nBlocks, nThreads>>>();

                    // start CUDA timer 
                    cudaEventRecord(start);
                        
                    // run kernel NUM_REPS times
                    if (strategy == 0){  // Simple kernel
                        for (int l=0; l<NUM_REPS; l++){
                            transposeSimple<<<nBlocks, nThreads>>>(dev_A, dev_A_T, tileDimension, blockRows);
                        } 
                    }
                    else if (strategy == 1){  // Coalesced kernel
                        for (int l=0; l<NUM_REPS; l++){
                            transposeCoalesced<<<nBlocks, nThreads, tileDimension * (tileDimension + 1) * sizeof(int)>>>(dev_A, dev_A_T, tileDimension, blockRows);
                        }
                    }
                    else if (strategy == 2){  // Diagonal kernel
                        for (int l=0; l<NUM_REPS; l++){
                            transposeDiagonal<<<nBlocks, nThreads, tileDimension * (tileDimension + 1) * sizeof(int)>>>(dev_A, dev_A_T, tileDimension, blockRows);
                        }
                    }
                    else {
                        throw runtime_error("Please choose 0, 1 or 2 for the strategy.");
                    }

                    // synchronize
                    cudaDeviceSynchronize();

                    // stop CUDA timer
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop); 

                    // Calculate elapsed time
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    // divide by NUM_REPS to get mean
                    milliseconds /= NUM_REPS;

                    // save execution time to file
				    myfile << milliseconds << ";";;

                    // copy back to host
                    cudaMemcpy(A_T, dev_A_T, N * sizeof(int), cudaMemcpyDeviceToHost);
                    
                    // Free timer events
                    cudaEventDestroy(start);
                    cudaEventDestroy(stop);

                    // free memory on device
                    cudaFree(dev_A);
                    cudaFree(dev_A_T);

                    // free memory on host
                    free(A);
                    free(A_T);

                    return 0;
                }
                myfile << "\n";  // new line in csv file
            }
            myfile << "\n";  // free line in csv file
        }
        // close file
		myfile.close();

    }
    else {
        int size = pow(2, atoi(argv[1]));
        int N = size * size;

        cout << "Size: " << size << endl;

		// call matrix generation with command line argument and receive matrix back
		int* A = generate_continous_matrix(size);

        for (int i=0; i<size; i++){
            for (int j=0; j<size; j++){
                cout << A[i * size + j] << "\t";
            }
            cout << endl;
        }
        cout << endl;

        // allocate memory on host
        int* A_T = (int*) malloc(N * sizeof(int));

        // determine kernel dimensions
        dim3 nBlocks (size / tileDimension, size / tileDimension, 1);
        dim3 nThreads (tileDimension, blockRows, 1);

        cout << "Blocks: " << size / tileDimension << " " << size / tileDimension << endl;
        cout << "Threads: " << tileDimension << " " << blockRows << endl;

        // allocate memory on device
        int *dev_A, *dev_A_T;

        cudaMalloc(&dev_A, N * sizeof(int));
        cudaMalloc(&dev_A_T, N * sizeof(int));

        // copy matrix to device
        cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

        // Create CUDA events to use for timing
	    cudaEvent_t start, stop;
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);

        // warmup to avoid timing startup TODO: is this necessary?
        warm_up_gpu<<<nBlocks, nThreads>>>();

        // start CUDA timer 
        cudaEventRecord(start);
            
        // run kernel
        if (strategy == 0){  // Simple kernel
            transposeSimple<<<nBlocks, nThreads>>>(dev_A, dev_A_T, tileDimension, blockRows);
        }
        else if (strategy == 1){  // Coalesced kernel
            transposeCoalesced<<<nBlocks, nThreads, tileDimension * (tileDimension + 1) * sizeof(int)>>>(dev_A, dev_A_T, tileDimension, blockRows);
        }
        else if (strategy == 2){  // Diagonal kernel
            transposeDiagonal<<<nBlocks, nThreads, tileDimension * (tileDimension + 1) * sizeof(int)>>>(dev_A, dev_A_T, tileDimension, blockRows);
        }
        else {
            throw runtime_error("Please choose 0, 1 or 2 for the strategy.");
        }

        // synchronize
        cudaDeviceSynchronize();

        // stop CUDA timer
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop); 

	    // Calculate elapsed time
	    float milliseconds = 0;
	    cudaEventElapsedTime(&milliseconds, start, stop);

	    printf("Kernel Time: %f ms\n", milliseconds);

        // copy back to host
        cudaMemcpy(A_T, dev_A_T, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        // display result
        for (int i=0; i<size; i++){
            for (int j=0; j<size; j++){
                cout << A_T[i*size + j] << "\t";
            }
            cout << endl;
        }

        // Free timer events
	    cudaEventDestroy(start);
	    cudaEventDestroy(stop);

        // free memory on device
        cudaFree(dev_A);
        cudaFree(dev_A_T);

        // free memory on host
        free(A);
        free(A_T);

        return 0;
    }

}