#include <iostream>
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
	// call matrix generation with command line argument and receive matrix back


	// define identity matrix in the meantime
	int A[4 * 4] = {1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,
					13, 14, 15, 16};

	// transpose matrix
	int* A_t = simple_transpose(A, 4);


	// check for correct transpose


	// display result
	for (int i=0; i<4; i++){
		for (int j=0; j<4; j++){
			cout << A_t[i*4 +j] << '\t';
		}
		cout << '\n';
	}

	return 0;
}