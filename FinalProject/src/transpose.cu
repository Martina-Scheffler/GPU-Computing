#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include "cublas_v2.h"


using namespace std;

#define NUM_REPS 10



__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
}


void transpose_cuSparse(){
    // load CSR matrix from file on host

    // create CSR CUDA matrix on device 
    cusparseCreateCsr();

    // transpose 

    // copy back
}

