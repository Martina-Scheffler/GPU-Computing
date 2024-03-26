#include <iostream>
using namespace std;

void square_transpose(int *A, int N){
	for (int i=0; i<N; i++){
		for (int j=i+1; j<N; j++){
			swap(A[N * i + j], A[j * N + i]);
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
	square_transpose(A, 4);


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