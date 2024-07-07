#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "include/import_sparse_matrix.h"

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

    // create CSR matrix using cuSparse
    cusparseSpMatDescr_t sparse_matrix;

    // void* for the three arrays
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
    
    // create CSR matrix
    cusparseCreateCsr(&sparse_matrix, rows, columns, nnz, dev_row_offsets, dev_col_indices, dev_values, CUDA_R_32I, 
                        CUDA_R_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // transpose
    
    // destroy matrix
    cusparseDestroySpMat(sparse_matrix);

    // free device memory
    cudaFree(dev_rows_offsets);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);
}


void transpose_cuSparse_COO(string file){
    // load COO matrix from file
    int rows, cols, nnz;
    int *row_indices, *col_indices;
    float* values;

    coo_from_file(file, rows, cols, nnz, row_indices, col_indices, values);

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
    cusparseCreateCoo(&sparse_matrix, rows, columns, nnz, dev_row_indices, dev_col_indices, dev_values, CUDA_R_32I, 
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // transpose 

    // copy back


    // destroy matrix
    cusparseDestroySpMat(sparse_matrix);

    // free device memory
    cudaFree(dev_rows_indices);
    cudaFree(dev_col_indices);
    cudaFree(dev_values);
}


int main(int argc, char* argv[]){
    transpose_cuSparse_COO("test_matrices/coo/1-bp_200_coo.csv")
    
    return 0;
}

