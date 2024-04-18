#include <iostream>
#include <cmath>
#include <chrono>
#include <map>
#include <fstream>

#include "../include/matrix_generation.h"
#include "../include/check_correctness.h"

using namespace std;

void square_transpose(int *A, int N){
	for (int i=0; i<N; i++){
		for (int j=i+1; j<N; j++){
			swap(A[N * i + j], A[j * N + i]);
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
		myfile.open("output/square_transpose_03.csv");

		for (int i=1; i<=12; i++){ // from (2^1 x 2^1) to (2^10 x 2^10) matrices
			for (int j=0; j<10; j++){ // run each size ten times to compensate fluctuations
				int size = pow(2, i);

				// generate random matrix
				int* A = generate_random_matrix(size);

				// get time before execution
				auto start = chrono::high_resolution_clock::now();

				// transpose matrix
				square_transpose(A, size);

				// get time after execution
				auto stop = chrono::high_resolution_clock::now();

				// calculate execution time
				const std::chrono::duration<double, std::milli> duration = stop - start;

				// save execution time to file
				myfile << duration.count();
				if (j != 9){
					myfile << ";";
				}

				free(A);
				
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
		square_transpose(A, size);

		// Get time after execution
		auto stop = chrono::high_resolution_clock::now();

		// Calculate execution time
		const std::chrono::duration<double, std::milli> duration = stop - start;
		//auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

		// display execution time
		cout << "Execution Time: " << duration.count() << " ms" << endl;

		free(A);
	
	}

	return 0;
}