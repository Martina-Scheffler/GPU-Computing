#include <iostream>
#include <cmath>
#include <chrono>
#include <map>
#include <fstream>

#include "../include/matrix_generation.h"

using namespace std;

int BLOCK_SIZE = 32;

void square_transpose(int *A, int N){
	for (int i=0; i<N; i++){
		for (int j=i+1; j<N; j++){
			swap(A[N * i + j], A[j * N + i]);
		}
	}
}

void block_transpose(int *A, int n, int N){
	if (n <= BLOCK_SIZE){
		// square_transpose(A, n); todo: find out why this is not working
		for (int i = 0; i < n; i++){
            for (int j = 0; j < i; j++){
                swap(A[i * N + j], A[j * N + i]);
			}
		}
	}
	else {
		int k = n / 2;  // divide matrix into blocks of n/2 x n/2

		// todo: N or n in lower blocks?
		block_transpose(A, k, N);  // upper left
		block_transpose(A + k, k, N);  // upper right
		block_transpose(A + k * N, k, N);  // lower left
		block_transpose(A + k * N + k, k, N);  // lower right

		// swap blocks
		for (int i=0; i<k; i++){
			for (int j=0; j<k; j++){
				swap(A[i * N + (j + k)], A[(i + k) * N + j]);
			}
		}
	}
}


int main (int argc, char* argv[]){
	// check if the matrix size was provided
	if (argc < 2){
		throw runtime_error("Please enter an integer N as argument to generate a matrix of size 2^N x 2^N.");
	}

	if (atoi(argv[1]) == 0){ // use zero as a key to run tests for the paper
		// open file to store execution times
		std::ofstream myfile;
		string extension = argv[2]; // append extension to save output to the correct file
		myfile.open("output/block_transpose_" + extension + ".csv");

		for (int i=1; i<=12; i++){ // from (2^1 x 2^1) to (2^10 x 2^10) matrices
			for (int j=0; j<10; j++){ // run each size ten times to compensate fluctuations
				int size = pow(2, i);

				// generate random matrix
				int* A = generate_random_matrix(size);

				// get time before execution
				auto start = chrono::high_resolution_clock::now();

				// transpose matrix
				block_transpose(A, size, size);

				// get time after execution
				auto stop = chrono::high_resolution_clock::now();

				// calculate execution time
				const std::chrono::duration<double, std::milli> duration = stop - start;

				// save execution time to file
				myfile << duration.count();
				if (j != 9){
					myfile << ";";
				}
				
			}

			// next line
			myfile << "\n";

		}
		// close file
		myfile.close();

	}
	else {
		int size = pow(2, atoi(argv[1]));

		// call matrix generation with command line argument and receive matrix back
		int* A = generate_continous_matrix(size);

		// Get time before execution
		auto start = chrono::high_resolution_clock::now();

		// transpose matrix
		block_transpose(A, size, size);

		// Get time after execution
		auto stop = chrono::high_resolution_clock::now();

		// Calculate execution time
		const std::chrono::duration<double, std::milli> duration = stop - start;
		//auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

		// display execution time
		cout << "Execution Time: " << duration.count() << " ms" << endl;
	
	}

	return 0;
}