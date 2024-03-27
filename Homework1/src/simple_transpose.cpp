#include <iostream>
#include <cmath>

#include "../include/matrix_generation.h"
#include "../include/check_correctness.h"

using namespace std;

int* simple_transpose(int *A, int N){
	int* A_t = (int*) malloc(N * N * sizeof(int));

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++){
			A_t[N * i + j] = A[j * N + i];
		}
	}

	return A_t;
}


int main (int argc, char* argv[]){
	// check if the matrix size was provided
	if (argc < 2){
		throw runtime_error("Please enter an integer N as argument to generate a matrix of size 2^N x 2^N.");
	}

	int size = pow(2, atoi(argv[1]));

	// call matrix generation with command line argument and receive matrix back
	int* A = generate_continous_matrix(size);

	// transpose matrix
	int* A_t = simple_transpose(A, size);

	// check for correctness using Eigen
	if (!check_transpose_correctness(A, A_t, size)){
		throw runtime_error("Checking for correctness failed.");
	}

	// display result
	for (int i=0; i<4; i++){
		for (int j=0; j<4; j++){
			cout << A_t[i*4 +j] << '\t';
		}
		cout << '\n';
	}

	return 0;
}