#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "../include/import_sparse_matrix.h"

using namespace std;

#define NUM_REPS 100


__global__ void transpose_COO(int* row_indices, int* column_indices, int* row_indices_tp, int* col_indices_tp, int nnz){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    

    while (idx < nnz){
        // swap row and columns        
        row_indices_tp[idx] = column_indices[idx];
        col_indices_tp[idx] = row_indices[idx];        
        
        idx += gridDim.x * blockDim.x;
    }
}


__global__ void CSR2COO(int* row_offsets_csr, int* row_indices_coo, int rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements_in_row;

    while (idx < rows){
        // recreate row indices from offset
        num_elements_in_row = row_offsets_csr[idx + 1] - row_offsets_csr[idx];

        for (int i=0; i<num_elements_in_row; i++){
            row_indices_coo[row_offsets_csr[idx] + i] = idx;
        }

        idx += gridDim.x * blockDim.x;
    }
}


__global__ void COO2CSR(int rows, int nnz, int* num_elements_in_row, int* saved_values_in_row,
                        int* row_indices_coo, int* col_indices_coo, float* values_coo,
                        int* row_offsets_csr, int* col_indices_csr, float* values_csr){
    int original_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int idx = original_idx;

    // count number of values in rows
    while (idx < rows){
        num_elements_in_row[idx] = 0;
        for (int i=0; i<nnz; i++){
            if (row_indices_coo[i] == idx){
                num_elements_in_row[idx]++;                
            }
        }

        idx += offset;
    }

    __syncthreads();  // wait for all threads to finish

    // sum them up
    idx = original_idx;
    while (idx < rows){
        row_offsets_csr[idx+1] = 0;
        for (int i=0; i<idx+1; i++){
            row_offsets_csr[idx+1] += num_elements_in_row[i];
        }

        idx += offset;
    }

    __syncthreads();  // wait for all threads to finish

    // figure out where the columns and values go
    idx = original_idx;
    while (idx < rows){
        saved_values_in_row[idx] = 0;
        for (int i=0; i<nnz; i++){
            if (row_indices_coo[i] == idx){
                col_indices_csr[row_offsets_csr[idx] + saved_values_in_row[idx]] = col_indices_coo[i];
                values_csr[row_offsets_csr[idx] + saved_values_in_row[idx]] = values_coo[i];
                saved_values_in_row[idx]++;
            }
        }
        idx += offset;
    }
}

__global__ void CSR2CSC(int rows, int columns, int nnz, int* num_elements_in_col, 
                        int* row_offsets_csr, int* column_indices_csr, float* values_csr,
                        int* row_indices_csc, int* column_offsets_csc,  float* values_csc, 
                        int* saved_values_in_col){
    int original_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int idx = original_idx;

    // count number of non-zero elements per column
    while (idx < columns){
        num_elements_in_col[idx] = 0;
        for (int i=0; i<nnz; i++){
            if (column_indices_csr[i] == idx){
                num_elements_in_col[idx]++;
            }
        }
        idx += offset;
    }

    __syncthreads();

    // sum up the values to find column offsets
    idx = original_idx;
    while (idx < columns){
        column_offsets_csc[idx+1] = 0;
        for (int i=0; i<idx+1; i++){
            column_offsets_csc[idx+1] += num_elements_in_col[i];
        }
        idx += offset;
    }

    __syncthreads();

    // insert row indices and values in the correct order
    idx = original_idx;
    int num_values;

    while (idx < columns){
        saved_values_in_col[idx] = 0;

        for (int i=0; i<rows; i++){
            num_values = row_offsets_csr[i+1] - row_offsets_csr[i];

            for (int j=0; j<num_values; j++){
                if (column_indices_csr[row_offsets_csr[i] + j] == idx){
                    row_indices_csc[column_offsets_csc[idx] + saved_values_in_col[idx]] = i;
                    values_csc[column_offsets_csc[idx] + saved_values_in_col[idx]] = values_csr[row_offsets_csr[i] + j];
                    saved_values_in_col[idx]++;
                }
            }
        }
        idx += offset;
    }
}


void transpose_own_CSR(string file, string timing_file){
    // file to save execution time for bandwidth analysis
    std::ofstream myfile;
	myfile.open(timing_file);

    // load CSR matrix from file
    int rows, columns, nnz;
    int *row_offsets, *col_indices;
    float* values;

    csr_from_file(file, rows, columns, nnz, row_offsets, col_indices, values);

    // create arrays on device
    int *dev_row_offsets_csr, *dev_col_indices_csr;
    float* dev_values_csr;

    // allocate memory on device
    cudaMalloc(&dev_row_offsets_csr, (rows+1) * sizeof(int));
    cudaMalloc(&dev_col_indices_csr, nnz * sizeof(int));
    cudaMalloc(&dev_values_csr,  nnz * sizeof(float));

    // copy entries to device
    cudaMemcpy(dev_row_offsets_csr, row_offsets, (rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices_csr, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values_csr, values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory on device
    int *dev_num_elements_in_col, *dev_saved_values_in_col, *dev_column_offsets_csc, *dev_row_indices_csc;
    float* dev_values_csc;

    cudaMalloc(&dev_num_elements_in_col, columns * sizeof(int));
    cudaMalloc(&dev_saved_values_in_col, columns * sizeof(int));
    cudaMalloc(&dev_column_offsets_csc, (columns+1) * sizeof(int));
    cudaMalloc(&dev_row_indices_csc, nnz * sizeof(int));
    cudaMalloc(&dev_values_csc, nnz * sizeof(float));

    // create blocks and threads
    dim3 nBlocks(1, 1, 1);
    dim3 nThreads(1024, 1, 1);

    // Create CUDA events to use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0;

    // start CUDA timer 
    cudaEventRecord(start, 0);

    // invoke kernels NUM_REPS times 
    for (int k=0; k<NUM_REPS; k++){
        CSR2CSC<<<nBlocks, nThreads>>>(rows, columns, nnz, dev_num_elements_in_col, dev_row_offsets_csr, 
                                        dev_col_indices_csr, dev_values_csr, dev_row_indices_csc, dev_column_offsets_csc,
                                        dev_values_csc, dev_saved_values_in_col);
        
        // synchronize
        cudaDeviceSynchronize();
    }

    // stop CUDA timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 

    // Calculate elapsed time
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // divide by NUM_REPS to get mean
    milliseconds /= NUM_REPS;

    // save execution time and configuration to file
    myfile << milliseconds << "\n";
    myfile << rows << "\n";
    myfile << columns << "\n";
    myfile << nnz << "\n";

    // copy results back to host
    int *row_offsets_tp = (int*) malloc((columns+1) * sizeof(int));
    int *col_indices_tp = (int*) malloc(nnz * sizeof(int));
    float* values_tp = (float*) malloc(nnz * sizeof(float));

    cudaMemcpy(col_indices_tp, dev_row_indices_csc, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offsets_tp, dev_column_offsets_csc, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values_tp, dev_values_csc, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // save transposed matrix to file
    transposed_csr_to_file(file, columns, rows, nnz, row_offsets_tp, col_indices_tp, values_tp);

    // free device memory
    cudaFree(dev_row_offsets_csr);
    cudaFree(dev_col_indices_csr);
    cudaFree(dev_values_csr);
    cudaFree(dev_num_elements_in_col);
    cudaFree(dev_saved_values_in_col);
    cudaFree(dev_column_offsets_csc);
    cudaFree(dev_row_indices_csc);
    cudaFree(dev_values_csc);

    // free timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free host memory
    free(row_offsets);
    free(col_indices);
    free(values);
    free(row_offsets_tp);
    free(col_indices_tp);
    free(values_tp);

    // close file
    myfile.close();
}



void transpose_own_COO(string file, string timing_file){
    // file to save execution time for bandwidth analysis
    std::ofstream myfile;
	myfile.open(timing_file);

    // load COO from file
    int rows, columns, nnz;
    int *row_indices, *col_indices;
    float* values;

    coo_from_file(file, rows, columns, nnz, row_indices, col_indices, values);

    // create arrays on device
    int *dev_row_indices, *dev_col_indices;
    int *dev_row_indices_tp, *dev_col_indices_tp;

    // allocate memory on device
    cudaMalloc(&dev_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));
    cudaMalloc(&dev_row_indices_tp, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices_tp, nnz * sizeof(int));


    // copy entries to device
    cudaMemcpy(dev_row_indices, row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events to use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // try different grid and block sizes and find fastest
    float min_time = INFINITY;
    int min_blocks;
    int min_threads;
    int possible_blocks = ceil(nnz / 1024.);
    int nBlocks;
    int nThreads;
   
    float milliseconds;
    for (int i=1; i<=possible_blocks; i++){
        nBlocks = i; 

        if (i == 1){
            // test diferent numbers of threads
            for (int j=2; j<=1024; j*=2){
                nThreads = j;

                // start CUDA timer 
                cudaEventRecord(start, 0);

                // invoke kernel NUM_REPS times 
                for (int k=0; k<NUM_REPS; k++){
                    transpose_COO<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
                }

                // synchronize
                cudaDeviceSynchronize();

                // stop CUDA timer
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop); 

                // Calculate elapsed time
                milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);

                // divide by NUM_REPS to get mean
                milliseconds /= NUM_REPS;

                if (milliseconds < min_time){
                    min_time = milliseconds;
                    min_blocks = i;
                    min_threads = j;
                }
            }
        }
        else {
            nThreads = 1024;

            // start CUDA timer 
            cudaEventRecord(start, 0);

            // invoke kernel NUM_REPS times 
            for (int k=0; k<NUM_REPS; k++){
                transpose_COO<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, dev_row_indices_tp, dev_col_indices_tp, nnz);
            }

            // synchronize
            cudaDeviceSynchronize();

            // stop CUDA timer
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop); 

            // Calculate elapsed time
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            // divide by NUM_REPS to get mean
            milliseconds /= NUM_REPS;

            if (milliseconds < min_time){
                min_time = milliseconds;
                min_blocks = i;
                min_threads = 1024;
            }
        }
    }
    // find best configuration
    printf("Best configuration: %d, %d\n", min_blocks, min_threads);

    // save execution time and configuration to file
    myfile << milliseconds << "\n";
    myfile << rows << "\n";
    myfile << columns << "\n";
    myfile << nnz << "\n";
    myfile << min_blocks << "\n";
    myfile << min_threads << "\n";

    // copy back
    cudaMemcpy(row_indices, dev_row_indices_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices, dev_col_indices_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // save result to file
    transposed_coo_to_file(file, columns, rows, nnz, row_indices, col_indices, values);

    // close file
	myfile.close();

    // free device memory
    cudaFree(dev_row_indices);
    cudaFree(dev_col_indices);

    // free host memory
    free(row_indices);
    free(col_indices);
    free(values);

    // free timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void transpose_own_via_COO(string file, string timing_file, bool find_best_config){
    // file to save execution time for bandwidth analysis
    std::ofstream myfile;
	myfile.open(timing_file);

    // load CSR matrix from file
    int rows, columns, nnz;
    int *row_offsets, *col_indices;
    float* values;

    csr_from_file(file, rows, columns, nnz, row_offsets, col_indices, values);

    // create arrays on device
    int *dev_row_offsets, *dev_col_indices;
    float* dev_values;

    // allocate memory and copy to device
    cudaMalloc(&dev_row_offsets, (rows+1) * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));
    cudaMalloc(&dev_values,  nnz * sizeof(float));

    cudaMemcpy(dev_row_offsets, row_offsets, (rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory for COO
    int *dev_row_indices;
    cudaMalloc(&dev_row_indices, nnz * sizeof(int));

    // device variables for COO transpose
    int *dev_tp_row_indices, *dev_tp_col_indices;
    cudaMalloc(&dev_tp_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_tp_col_indices, nnz * sizeof(int));

    // device variables for COO->CSR conversion
    int *dev_num_elements_in_row, *dev_saved_values_in_row;
    cudaMalloc(&dev_num_elements_in_row, rows * sizeof(int));
    cudaMalloc(&dev_saved_values_in_row, rows * sizeof(int));

    int* dev_row_offsets_tp, *dev_col_indices_tp;
    float* dev_values_tp;

    cudaMalloc(&dev_row_offsets_tp, (columns+1) * sizeof(int));
    cudaMalloc(&dev_col_indices_tp, nnz * sizeof(int));
    cudaMalloc(&dev_values_tp, nnz * sizeof(float));

    // Create CUDA events to use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds_overall = 0.0;
    float milliseconds = 0.0;

    float min_time = INFINITY;
    int min_blocks;
    int min_threads;

    // create blocks and threads
    int possible_blocks = ceil(nnz / 1024.);

    if (find_best_config){
        // invoke CSR2COO kernel NUM_REPS times and find best config
        for (int i=1; i<=possible_blocks; i++){
            if (i == 1){
                for (int j=2; j<=1024; j*=2){
                    // start CUDA timer 
                    cudaEventRecord(start, 0);

                    for (int k=0; k<NUM_REPS; k++){
                        CSR2COO<<<i, j>>>(dev_row_offsets, dev_row_indices, rows);
                        cudaDeviceSynchronize();
                    }

                    // stop CUDA timer
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop); 

                    // Calculate elapsed time
                    cudaEventElapsedTime(&milliseconds, start, stop);

                    // divide by NUM_REPS to get mean
                    milliseconds /= NUM_REPS;

                    if (milliseconds < min_time){
                        min_time = milliseconds;
                        min_blocks = i;
                        min_threads = j;
                    }
                }
            }
            else {
                // start CUDA timer 
                cudaEventRecord(start, 0);

                for (int k=0; k<NUM_REPS; k++){
                    CSR2COO<<<i, 1024>>>(dev_row_offsets, dev_row_indices, rows);
                    cudaDeviceSynchronize();
                }

                // stop CUDA timer
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop); 

                // Calculate elapsed time
                cudaEventElapsedTime(&milliseconds, start, stop);

                // divide by NUM_REPS to get mean
                milliseconds /= NUM_REPS;

                if (milliseconds < min_time){
                    min_time = milliseconds;
                    min_blocks = i;
                    min_threads = 1024;
                }
            }
        }

        printf("Best config for COO-CSR: %d, %d\n", min_blocks, min_threads);
        milliseconds_overall += min_time;
    }
    else {
        // start CUDA timer 
        cudaEventRecord(start, 0);

        for (int k=0; k<NUM_REPS; k++){
            CSR2COO<<<possible_blocks, 1024>>>(dev_row_offsets, dev_row_indices, rows);
            cudaDeviceSynchronize();
        }

        // stop CUDA timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 

        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);

        // divide by NUM_REPS to get mean
        milliseconds /= NUM_REPS;
        milliseconds_overall += milliseconds;
    }

    printf("Best config for COO-CSR: %d, %d\n", min_blocks, min_threads);
    milliseconds_overall += min_time;

    // invoke COO transpose kernel NUM_REPS times
    // start CUDA timer 
    cudaEventRecord(start, 0);

    for (int k=0; k<NUM_REPS; k++){
        transpose_COO<<<possible_blocks, 1024>>>(dev_row_indices, dev_col_indices, dev_tp_row_indices, dev_tp_col_indices, nnz);
        cudaDeviceSynchronize();
    }

    // stop CUDA timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    // divide by NUM_REPS to get mean
    milliseconds /= NUM_REPS;

    milliseconds_overall += milliseconds;

    min_time = INFINITY;

    // invoke COO-CSR kernel NUM_REPS times and find best number of threads
    for (int i=2; i<=1024; i*=2){
        // start CUDA timer 
        cudaEventRecord(start, 0);

        for (int k=0; k<NUM_REPS; k++){
            COO2CSR<<<1, i>>>(rows, nnz, dev_num_elements_in_row, dev_saved_values_in_row,
                                    dev_tp_row_indices, dev_tp_col_indices, dev_values,
                                    dev_row_offsets_tp, dev_col_indices_tp, dev_values_tp);
                            
            cudaDeviceSynchronize();
        }

        // stop CUDA timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); 

        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);

        // divide by NUM_REPS to get mean
        milliseconds /= NUM_REPS;

        if (milliseconds < min_time){
            min_time = milliseconds;
            min_threads = i;
        }
    }

    printf("Best config for COO-CSR: %d\n", min_threads);
    milliseconds_overall += min_time;

    // save execution time and configuration to file
    myfile << milliseconds_overall << "\n";
    myfile << rows << "\n";
    myfile << columns << "\n";
    myfile << nnz << "\n";

    // copy back
    cudaMemcpy(row_offsets, dev_row_offsets_tp, (columns+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices, dev_col_indices_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, dev_values_tp, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // save COO matrix
    transposed_csr_to_file(file, columns, rows, nnz, row_offsets, col_indices, values);


    // free device memory
    cudaFree(dev_row_offsets);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);
    cudaFree(dev_row_indices);
    cudaFree(dev_tp_row_indices);
    cudaFree(dev_tp_col_indices);
    cudaFree(dev_num_elements_in_row);
    cudaFree(dev_saved_values_in_row);
    cudaFree(dev_row_offsets_tp);
    cudaFree(dev_col_indices_tp);
    cudaFree(dev_values_tp);

    // free timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free host memory
    free(row_offsets);
    free(col_indices);
    free(values);

    // close file
    myfile.close();
}


void transpose_cuSparse_CSR(string file, string timing_file){
    // file to save execution time for bandwidth analysis
    std::ofstream myfile;
	myfile.open(timing_file);

    // load CSR matrix from file
    int rows, columns, nnz;
    int *row_offsets, *col_indices;
    float* values;

    csr_from_file(file, rows, columns, nnz, row_offsets, col_indices, values);

    // create arrays on device
    int *dev_row_offsets, *dev_col_indices;
    float* dev_values;

    // allocate memory on device
    cudaMalloc(&dev_row_offsets, (rows+1) * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));
    cudaMalloc(&dev_values,  nnz * sizeof(float));

    // copy entries to device
    cudaMemcpy(dev_row_offsets, row_offsets, (rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events to use for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // reserve buffer space necessary for the transpose
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int *dev_tp_row_indices, *dev_tp_col_offsets;
    float* dev_tp_values;
    cudaMalloc(&dev_tp_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_tp_col_offsets, (columns+1) * sizeof(int));
    cudaMalloc(&dev_tp_values, nnz * sizeof(float));

    size_t buffer_size;
    cusparseCsr2cscEx2_bufferSize(handle, rows, columns, nnz, dev_values, dev_row_offsets, dev_col_indices, 
                                    dev_tp_values, dev_tp_col_offsets, dev_tp_row_indices, CUDA_R_32F, 
                                    CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                                    &buffer_size); 

    void* buffer;
    cudaMalloc(&buffer, buffer_size);

    // start CUDA timer 
    cudaEventRecord(start, 0);

    // run NUM_REPS times
    for (int i=0; i<NUM_REPS; i++){
        // transpose by converting from CSR to CSC
        cusparseCsr2cscEx2(handle, rows, columns, nnz, dev_values, dev_row_offsets, dev_col_indices, dev_tp_values, 
                            dev_tp_col_offsets, dev_tp_row_indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    }

    // stop CUDA timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // divide by NUM_REPS to get mean
    milliseconds /= NUM_REPS;

    // save execution time and buffer size to file
    myfile << milliseconds << "\n";
    myfile << rows << "\n";
    myfile << columns << "\n";
    myfile << nnz << "\n";
    myfile << buffer_size << "\n";

    // copy results back to host
    int *row_offsets_tp = (int*) malloc((columns+1) * sizeof(int));
    int *col_indices_tp = (int*) malloc(nnz * sizeof(int));
    float* values_tp = (float*) malloc(nnz * sizeof(float));

    cudaMemcpy(col_indices_tp, dev_tp_row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offsets_tp, dev_tp_col_offsets, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values_tp, dev_tp_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // save transposed matrix to file
    transposed_csr_to_file(file, columns, rows, nnz, row_offsets_tp, col_indices_tp, values_tp);

    // close file
	myfile.close();

    // free timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // destroy handle
    cusparseDestroy(handle);

    // free device memory
    cudaFree(dev_row_offsets);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);

    cudaFree(buffer);

    cudaFree(dev_tp_row_indices);
    cudaFree(dev_tp_col_offsets);
    cudaFree(dev_tp_values);

    // free host memory
    free(row_offsets);
    free(col_indices);
    free(values);
    free(row_offsets_tp);
    free(col_indices_tp);
    free(values_tp);
}


void transpose_cuSparse_COO(string file){
    // load COO matrix from file
    int rows, columns, nnz;
    int *row_indices, *col_indices;
    float* values;

    coo_from_file(file, rows, columns, nnz, row_indices, col_indices, values);

    // create COO CUDA matrix using cuSparse 
    cusparseSpMatDescr_t sparse_matrix;

    // void* for the three arrays
    int *dev_row_indices, *dev_col_indices;
    float* dev_values;

    // allocate memory on device
    cudaMalloc(&dev_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));
    cudaMalloc(&dev_values,  nnz * sizeof(float));

    // copy entries to device
    cudaMemcpy(dev_row_indices, row_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // create COO matrix
    cusparseCreateCoo(&sparse_matrix, rows, columns, nnz, dev_row_indices, dev_col_indices, dev_values, CUSPARSE_INDEX_32I, 
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // reserve necessary buffer space
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    const int alpha = 1;
    const int beta = 0;
    cusparseDnMatDescr_t dense_matrix;
    float* dev_dmat_values;
    cudaMalloc(&dev_dmat_values, rows * columns * sizeof(float));
    cusparseCreateDnMat(&dense_matrix, rows, columns, rows, dev_dmat_values, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    size_t buffer_size;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, 
                            sparse_matrix, dense_matrix, &beta, dense_matrix, CUDA_R_32F, CUSPARSE_SPMM_COO_ALG1, 
                            &buffer_size);
    void* buffer;
    cudaMalloc(&buffer, buffer_size);

    // transpose
    cusparseSpMM(handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse_matrix, 
                    dense_matrix, &beta, dense_matrix, CUDA_R_32F, CUSPARSE_SPMM_COO_ALG1, buffer);

    // save values back into sparse matrix
    cusparseDenseToSparse_bufferSize(handle, dense_matrix, sparse_matrix, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size);
    void* buffer_convert;
    cudaMalloc(&buffer_convert, buffer_size);
    cusparseDenseToSparse_analysis(handle, dense_matrix, sparse_matrix, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_convert);
    cusparseDenseToSparse_convert(handle, dense_matrix, sparse_matrix, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buffer_convert);

    // copy back to host
    cusparseIndexType_t index_type = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType data_type = CUDA_R_32F;
    cusparseCooGet(sparse_matrix, (int64_t*)&rows, (int64_t*)&columns, (int64_t*)&nnz, 
                    (void **)&dev_row_indices, (void **)&dev_col_indices, (void **)&dev_values, 
                    &index_type, &index_base, &data_type);

    cudaMemcpy(row_indices, dev_row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices, dev_col_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, dev_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);
       
    // write transposed matrix to file
    transposed_coo_to_file(file, columns, rows, nnz, row_indices, col_indices, values);                            

    // destroy matrix
    cusparseDestroySpMat(sparse_matrix);
    cusparseDestroyDnMat(dense_matrix);

    // destroy handle
    cusparseDestroy(handle);

    // free device memory
    cudaFree(dev_row_indices);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);
    cudaFree(buffer);
    cudaFree(dev_dmat_values);
    cudaFree(buffer_convert);

    // free host memory 
    free(row_indices);
    free(col_indices);
    free(values);
}


int main(int argc, char* argv[]){
    printf("Started script\n");
    if (argc < 2){
        throw runtime_error("Please choose a strategy");
    }

    // Strategy 0: cuSPARSE CSR
    if (atoi(argv[1]) == 0){
        printf("Use CSR format and the cuSPARSE library.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        string argv2 = argv[2];
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_cuSparse_CSR("test_matrices/csr/" + to_string(i) + "_csr.csv", 
                                        "output/csr_cusparse_" + to_string(i) + ".csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_cuSparse_CSR("test_matrices/csr/" + to_string(atoi(argv[2])) + "_csr.csv",
                                    "output/csr_cusparse_" + to_string(atoi(argv[2])) + ".csv");
        }
    }

    // Strategy 1: own kernel CSR via CSR-CSC conversion
    if (atoi(argv[1]) == 1){
        printf("Use CSR format via CSR-CSC conversion.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        string argv2 = argv[2];
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_own_CSR("test_matrices/csr/" + to_string(i) + "_csr.csv",
                                    "output/csr_csc_own_" + to_string(i) + ".csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_own_CSR("test_matrices/csr/" + to_string(atoi(argv[2])) + "_csr.csv",
                                "output/csr_csc_own_" + to_string(atoi(argv[2])) + ".csv");
        }
    }

    // Strategy 2: own COO transpose kernel
    if (atoi(argv[1]) == 2){
        printf("Use COO format and own kernel.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        string argv2 = argv[2];
        
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_own_COO("test_matrices/coo/" + to_string(i) + "_coo.csv",
                                    "output/coo_own_" + to_string(i) + ".csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_own_COO("test_matrices/coo/" + to_string(atoi(argv[2])) + "_coo.csv",
                                "output/coo_own_" + to_string(atoi(argv[2])) + ".csv");
        }
        
    }

    // Strategy 3: own kernel CSR via COO
    if (atoi(argv[1]) == 3){
        printf("Use CSR format and transpose via own COO kernel.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        bool find_best_config = false;
        if (argc == 4){
            if (atoi(argv[3]) == 1){
                find_best_config = true;
            }
        }

        string argv2 = argv[2];
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_own_via_COO("test_matrices/csr/" + to_string(i) + "_csr.csv",
                                    "output/csr_coo_own_" + to_string(i) + ".csv", find_best_config);
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_own_via_COO("test_matrices/csr/" + to_string(atoi(argv[2])) + "_csr.csv",
                                "output/csr_coo_own_" + to_string(atoi(argv[2])) + ".csv", find_best_config);
        }
        
    }


    
    return 0;
}

