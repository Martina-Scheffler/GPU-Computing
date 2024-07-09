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


void transpose_cuSparse_CSR(string file){
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
                              
    // transpose by converting from CSR to CSC
    void* buffer;
    cudaMalloc(&buffer, buffer_size);
    cusparseCsr2cscEx2(handle, rows, columns, nnz, dev_values, dev_row_offsets, dev_col_indices, dev_tp_values, 
                        dev_tp_col_offsets, dev_tp_row_indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);

    // copy results back to host
    int *row_offsets_tp = (int*) malloc((columns+1) * sizeof(int));
    int *col_indices_tp = (int*) malloc(nnz * sizeof(int));
    float* values_tp = (float*) malloc(nnz * sizeof(float));

    cudaMemcpy(col_indices_tp, dev_tp_row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offsets_tp, dev_tp_col_offsets, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values_tp, dev_tp_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // save transposed matrix to file
    transposed_csr_to_file(file, columns, rows, nnz, row_offsets_tp, col_indices_tp, values_tp);

    // destroy handle
    cusparseDestroy(handle);

    // free device memory
    cudaFree(dev_row_offsets);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);

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
    
    // transpose 

    // copy back


    // destroy matrix
    cusparseDestroySpMat(sparse_matrix);

    // free device memory
    cudaFree(dev_row_indices);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);

    // free host memory 
    free(row_indices);
    free(col_indices);
    free(values);
}


int main(int argc, char* argv[]){
    if (argc < 2){
        throw runtime_error("Please choose a strategy");
    }
    if (atoi(argv[1]) == 0){
        printf("Use CSR format and the cuSPARSE library.\n");

        // check which test matrix to use
        if (argc < 3){
            throw runtime_error("Please choose a test matrix");
        }

        if (to_string(argv[2]) == "all"){
            for (int i=1; i<11; i++){
                printf("Transposing matrix %d\n", i);
                transpose_cuSparse_CSR("test_matrices/csr/" + i + "_csr.csv");
            }
        }
        else {
            printf("Transposing matrix %d\n", atoi(argv[2]));
            transpose_cuSparse_CSR("test_matrices/csr/" + atoi(argv[2]) + "_csr.csv");
        }
    }
    
    return 0;
}

