#include <iostream>
using namespace std;

int BLOCK_SIZE = 2;

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
	// call matrix generation with command line argument and receive matrix back


	// define identity matrix in the meantime
	int A[4 * 4] = {1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,
					13, 14, 15, 16};

	// transpose matrix
	block_transpose(A, 4, 4);


	// check for correct transpose


	// display result
	for (int i=0; i<4; i++){
		for (int j=0; j<4; j++){
			cout << A[i*4 +j] << '\t';
		}
		cout << '\n';
	}

	return 0;
}