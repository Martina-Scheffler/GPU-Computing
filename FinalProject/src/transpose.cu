#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "../include/import_sparse_matrix.h"

using namespace std;

#define NUM_REPS 10



__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
}


__global__ void transpose_COO(int* row_indices, int* column_indices, int nnz){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp;

    while (idx < nnz){
        // swap row and columns
        //printf("%d: %d, %d\n", idx, row_indices[idx], column_indices[idx]);
        tmp = row_indices[idx];
        row_indices[idx] = column_indices[idx];
        column_indices[idx] = tmp;
        //printf("%d: %d, %d\n", idx, row_indices[idx], column_indices[idx]);
        
        idx += gridDim.x * blockDim.x;
    }
}


__global__ void CSR2COO(int* row_offsets, int* row_indices, int rows){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements_in_row;

    while (idx < rows){
        // recreate row indices from offset
        num_elements_in_row = row_offsets[idx + 1] - row_offsets[idx];

        for (int i=0; i<num_elements_in_row; i++){
            row_indices[row_offsets[idx] + i] = idx;
        }

        idx += blockDim.x;
    }
}

__global__ void CSR2CSC(int rows, int columns, int nnz, int* num_elements_in_col, 
                        int* row_offsets_csr, int* column_indices_csr, float* values_csr,
                        int* row_indices_csc, int* column_offsets_csc,  float* values_csc, 
                        int* values_stored_from_col){
    int original_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = original_idx;

    // count number of non-zero elements per column
    while (idx < nnz){
        num_elements_in_col[column_indices_csr[idx]] += 1;
        idx += blockDim.x;
    }

    __syncthreads();

    // sum up the values to find column offsets
    idx = original_idx;
    while (idx < columns){
        for (int i=0; i<idx+1; i++){
            column_offsets_csc[idx + 1] += num_elements_in_col[i];
        }
        idx += blockDim.x;
    }

    __syncthreads();

    // insert row indices and values in the correct order
    idx = original_idx;
    int num_values;
    int col;
    
    while (idx < rows){
        num_values = row_offsets_csr[idx+1] - row_offsets_csr[idx];
        for (int i=0; i<num_values; i++){
            col = column_indices_csr[row_offsets_csr[idx] + i];
            row_indices_csc[column_offsets_csc[col] + values_stored_from_col[col]] = idx;
            values_csc[column_offsets_csc[col] + values_stored_from_col[col]] = values_csr[row_offsets_csr[idx] + i];
            values_stored_from_col[col] += 1;
        }
        idx += blockDim.x;
    }
}


void transpose_own_CSR(string file, string output_file){
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

    // create necessary buffer arrays and copy to device
    int *num_elements_in_col = (int*) malloc(columns * sizeof(int));
    int *values_stored_from_col = (int*) malloc(columns * sizeof(int));
    int *column_offsets_csc = (int*) malloc((columns + 1) * sizeof(int));

    for (int i=0; i<columns; i++){
        num_elements_in_col[i] = 0;
        values_stored_from_col[i] = 0;
        column_offsets_csc[i] = 0;
    }
    column_offsets_csc[columns] = 0;

    // allocate memory on device
    int *dev_num_elements_in_col, *dev_values_stored_from_col, *dev_column_offsets_csc, *dev_row_indices_csc;
    float* dev_values_csc;

    cudaMalloc(&dev_num_elements_in_col, columns * sizeof(int));
    cudaMalloc(&dev_values_stored_from_col, columns * sizeof(int));
    cudaMalloc(&dev_column_offsets_csc, (columns+1) * sizeof(int));
    cudaMalloc(&dev_row_indices_csc, nnz * sizeof(int));
    cudaMalloc(&dev_values_csc, nnz * sizeof(float));

    // copy
    cudaMemcpy(dev_num_elements_in_col, num_elements_in_col, columns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_values_stored_from_col, values_stored_from_col, columns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_column_offsets_csc, column_offsets_csc, (columns+1) * sizeof(int), cudaMemcpyHostToDevice);

    // create blocks and threads
    dim3 nBlocks(1, 1, 1);
    dim3 nThreads(1024, 1, 1);

    // call kernel
    CSR2CSC<<<nBlocks, nThreads>>>(rows, columns, nnz, dev_num_elements_in_col, dev_row_offsets_csr, 
                                    dev_col_indices_csr, dev_values_csr, dev_row_indices_csc, dev_column_offsets_csc,
                                    dev_values_csc, dev_values_stored_from_col);
    
    // synchronize
    cudaDeviceSynchronize();

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
    cudaFree(dev_values_stored_from_col);
    cudaFree(dev_column_offsets_csc);
    cudaFree(dev_row_indices_csc);
    cudaFree(dev_values_csc);

    // free host memory
    free(row_offsets);
    free(col_indices);
    free(values);
    free(num_elements_in_col);
    free(values_stored_from_col);
    free(column_offsets_csc);    
    free(row_offsets_tp);
    free(col_indices_tp);
    free(values_tp);
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

    // allocate memory on device
    cudaMalloc(&dev_row_indices, nnz * sizeof(int));
    cudaMalloc(&dev_col_indices, nnz * sizeof(int));

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
    int possible_blocks = ceil(nnz / 1024);
    float milliseconds;
    for (int i=1; i<=possible_blocks; i++){
        dim3 nBlocks(i, 1, 1);

        if (i == 1){
            // test diferent numbers of threads
            for (int j=2; j<=1024; j*=2){
                dim3 nThreads(j, 1, 1);

                // start CUDA timer 
                cudaEventRecord(start, 0);

                // invoke kernel NUM_REPS times 
                for (int k=0; k<NUM_REPS; k++){
                    transpose_COO<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, nnz);
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
            dim3 nThreads(1024, 1, 1);

            // start CUDA timer 
            cudaEventRecord(start, 0);

            // invoke kernel NUM_REPS times 
            for (int k=0; k<NUM_REPS; k++){
                transpose_COO<<<nBlocks, nThreads>>>(dev_row_indices, dev_col_indices, nnz);
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
    cudaMemcpy(row_indices, dev_row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_indices, dev_col_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d, %d\n", row_indices[1], col_indices[1]);

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

    for (int i=0; i<nnz; i++){
        printf("%f\n", values);
    }
               
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

    // Strategy 1: cuSPARSE COO - Currently not working
    if (atoi(argv[1]) == 1){
        printf("Use COO format and the cuSPARSE library.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        string argv2 = argv[2];
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_cuSparse_COO("test_matrices/coo/" + to_string(i) + "_coo.csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_cuSparse_COO("test_matrices/coo/" + to_string(atoi(argv[2])) + "_coo.csv");
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
                                    "output/coo_own_" + to_string(atoi(argv[2])) + ".csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_own_COO("test_matrices/coo/" + to_string(atoi(argv[2])) + "_coo.csv",
                                "output/coo_own_" + to_string(atoi(argv[2])) + ".csv");
        }
        
    }

    // Strategy 3: own CSR2CSC kernel
    if (atoi(argv[1]) == 3){
        printf("Use CSR format and own CSR2CSC kernel.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        string argv2 = argv[2];
        if (argv2 == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_own_CSR("test_matrices/csr/" + to_string(i) + "_csr.csv",
                                    "output/csr_own_" + to_string(atoi(argv[2])) + ".csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_own_COO("test_matrices/csr/" + to_string(atoi(argv[2])) + "_csr.csv",
                                "output/csr_own_" + to_string(atoi(argv[2])) + ".csv");
        }
        
    }


    
    return 0;
}

