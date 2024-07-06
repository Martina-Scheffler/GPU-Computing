#ifndef IMPORT_SPARSE_MATRIX
#define IMPORT_SPARSE_MATRIX

void to_csr(double** dense_matrix, int M, int N, int nz, int* row_offsets, int* column_indices, double* values);
void csr_to_file(const char* file, int M, int N, int nz, int* row_offsets, int* column_indices, double* values);
void to_coo(double** dense_matrix, int M, int N, int nz, int* row_indices, int* column_indices, double* values);
void coo_to_file(const char* file, int M, int N, int nz, int* row_indices, int* column_indices, double* values);
void convert_mtx_to_file(const char* file);

#endif