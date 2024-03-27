#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;


bool check_transpose_correctness(int* matrix, int* transposed_matrix, int size){

	Map<Matrix<int, Dynamic, Dynamic, RowMajor> > A(matrix, size, size);
	Map<Matrix<int, Dynamic, Dynamic, RowMajor> > A_T(transposed_matrix, size, size);

	Matrix<int, Dynamic, Dynamic, RowMajor> eigen_transpose = A.transpose();

	if (eigen_transpose.isApprox(A_T)){
		return true;
	}
	else {
		return false;
	}
}