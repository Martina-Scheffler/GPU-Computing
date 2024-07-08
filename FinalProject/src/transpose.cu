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

    // for (int i=0; i<rows+1; i++){
    //     printf("%d\n", row_offsets[i]);
    // }

    // // create CSR matrix using cuSparse
    // cusparseSpMatDescr_t sparse_matrix;

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
    
    // create CSR matrix
    // cusparseCreateCsr(&sparse_matrix, rows, columns, nnz, dev_row_offsets, dev_col_indices, dev_values, 
    //                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

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

    cout << buffer_size << "\n"; 
                              
    // transpose by converting from CSR to CSC
    void* buffer;
    cudaMalloc(&buffer, buffer_size);
    cusparseStatus_t result = cusparseCsr2cscEx2(handle, rows, columns, nnz, dev_values, dev_row_offsets, dev_col_indices, dev_tp_values, 
                        dev_tp_col_offsets, dev_tp_row_indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);

    cout << result << "\n";

    // copy results back to host
    int *row_offsets_tp = (int*) malloc((columns+1) * sizeof(int));
    int *col_indices_tp = (int*) malloc(nnz * sizeof(int));
    float* values_tp = (float*) malloc(nnz * sizeof(float));

    cudaMemcpy(col_indices_tp, dev_tp_row_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offsets_tp, dev_tp_col_offsets, (columns + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values_tp, dev_tp_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<columns+1; i++){
        printf("%d\n", row_offsets_tp[i]);
    }

    // save transposed matrix to file
    transposed_csr_to_file(file, columns, rows, nnz, row_offsets_tp, col_indices_tp, values_tp);
    
    // // destroy matrix
    // cusparseDestroySpMat(sparse_matrix);

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
    cout << "Running cuSparse COO transpose" << "\n";
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
}


int main(int argc, char* argv[]){
    cout << "Matrix 0\n";
    transpose_cuSparse_CSR("test_matrices/csr/0-test_csr.csv");
    cout << "Matrix 1\n";
    transpose_cuSparse_CSR("test_matrices/csr/1-bp_200_csr.csv");
    cout << "Matrix 2\n";
    transpose_cuSparse_CSR("test_matrices/csr/2-fs_183-1_csr.csv");
    cout << "Matrix 3\n";
    transpose_cuSparse_CSR("test_matrices/csr/3-fs_541_1_csr.csv");
    cout << "Matrix 4\n";
    transpose_cuSparse_CSR("test_matrices/csr/4-pores_2_csr.csv");
    cout << "Matrix 5\n";
    transpose_cuSparse_CSR("test_matrices/csr/5-shl_200_csr.csv");
    cout << "Matrix 6\n";
    transpose_cuSparse_CSR("test_matrices/csr/6-GD96_a_csr.csv");
    cout << "Matrix 7\n";
    transpose_cuSparse_CSR("test_matrices/csr/7-GD00_c_csr.csv");
    cout << "Matrix 8\n";
    transpose_cuSparse_CSR("test_matrices/csr/8-ch5-5-b3_csr.csv");
    cout << "Matrix 9\n";
    transpose_cuSparse_CSR("test_matrices/csr/9-dw256A_csr.csv");
    cout << "Matrix 10\n";
    transpose_cuSparse_CSR("test_matrices/csr/10-qh768_csr.csv");
    
    return 0;
}

