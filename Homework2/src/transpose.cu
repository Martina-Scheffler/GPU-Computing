#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include "cublas_v2.h"


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

bool checkCorrectness(int* A, int* A_T, int size){
    // allocate memory on host
    float* res = (float*) malloc(size * size * sizeof(float));
    float* A_copy = (float*) malloc(size * size * sizeof(float));

    // allocate memory on device
    float *dev_A_check, *dev_A_T_check;
    cudaMalloc(&dev_A_check, size * size * sizeof(float));
    cudaMalloc(&dev_A_T_check, size * size * sizeof(float));

    // copy int array to float array
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
                A_copy[i * size + j] = (float) A[i * size + j];
        }
    }

    // copy to device
    cudaMemcpy(dev_A_check, A_copy, size * size * sizeof(float), cudaMemcpyHostToDevice);

    // transpose
    float const alpha(1.0);
    float const beta(0.0);
    cublasHandle_t handle;

    cublasCreate(&handle);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, &alpha, dev_A_check, size, &beta, dev_A_check, size, dev_A_T_check, size);
    cublasDestroy(handle);

    // copy back
    cudaMemcpy(res, dev_A_T_check, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    // check correctness
    bool correct = true;

    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            if (A_T[i * size + j] != (int) res[i * size + j]) {
                // printf("%d != %d\n", A_T[i * size + j], res[i * size + j]);
                correct = false;
            }
        }
    }

    // free memory on device
    cudaFree(dev_A_check);
    cudaFree(dev_A_T_check);

    // free memory on host
    free(A_copy);
    free(res);

    return correct;
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
    
    if (atoi(argv[1]) == 0){ // use zero for analyzing the effective bandwidth with different matrix sizes and parameters
        // open file to store execution times
		std::ofstream myfile;
		string extension = argv[2]; // append extension to save output to the correct file
		myfile.open("output/analyze_bandwidth_" + extension + ".csv");

        for (int i=2; i<=pow(2, 12); i*=2){  // from 2^1 to 2^12
            // matrix dimension
            int size = i; 
            int N = size * size;

            // loop over all possible values of tile dimension and block rows
            for (int j=2; j<=i; j*=2){  // 2 to i (matrix dimension)
                if (strategy == 0 && j > pow(2, 10)){  // maximum allowed number of threads limits j
                    break;
                }
                else if ((strategy == 1 || strategy == 2) && j >= 128){  // size of shared memory limits j
                    break;
                }

                tileDimension = j;

                for (int k=1; k<=j && (k*j<=1024) ; k*=2){  // 1 to j (tile dimension) and max. 1024 threads overall
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

                    // warmup to avoid timing startup 
                    warm_up_gpu<<<nBlocks, nThreads>>>();

                    // start CUDA timer 
                    cudaEventRecord(start, 0);
                        
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
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop); 

                    // copy back to host
                    cudaMemcpy(A_T, dev_A_T, N * sizeof(int), cudaMemcpyDeviceToHost);

                    // check correctness 
                    if (checkCorrectness(A, A_T, size)){
                        // Calculate elapsed time
                        float milliseconds = 0;
                        cudaEventElapsedTime(&milliseconds, start, stop);

                        // divide by NUM_REPS to get mean
                        milliseconds /= NUM_REPS;

                        // save execution time to file
                        myfile << milliseconds << ";";;
                    }
                    else {
                        // skip entry in file and print error (should not happen with defined TD and BR)
                        myfile << ";";
                        printf("ERROR | Size: %d, TD: %d, BR: %d \n", size, tileDimension, blockRows);
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
                }
                myfile << "\n";  // new line in csv file
            }
            myfile << "\n";  // free line in csv file
        }
        // close file
		myfile.close();

        return 0;

    }
    else {
        int size = pow(2, atoi(argv[1]));
        int N = size * size;

        cout << "Size: " << size << endl;

		// call matrix generation with command line argument and receive matrix back
		int* A = generate_continous_matrix(size);

        // print generated matrix - comment out for big matrices
        // for (int i=0; i<size; i++){
        //     for (int j=0; j<size; j++){
        //         cout << A[i * size + j] << "\t";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

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

        // warmup to avoid timing startup
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

        // check correctness
        if (!checkCorrectness(A, A_T, size)){
            printf("Incorrect Result!!!\n");
        }
        
        // display result - comment out for big matrices
        // for (int i=0; i<size; i++){
        //     for (int j=0; j<size; j++){
        //         cout << A_T[i*size + j] << "\t";
        //     }
        //     cout << endl;
        // }

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