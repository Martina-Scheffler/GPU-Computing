#include <fstream>
#include <algorithm> 
#include <iostream>
#include <cstdio>
#include <cmath> 

#include "mmio.h"

using namespace std;



inline bool isEqual(float x, float y)
{
  const float epsilon = 1e-5;
  return abs(x - y) <= epsilon * std::abs(x);
}


void to_csr(float** dense_matrix, int M, int N, int nz, int* row_offsets, int* column_indices, float* values){
    int nz_idx = 0;
    
    for (int i=0; i<M; i++){
        row_offsets[i] = nz_idx;
        for (int j=0; j<N; j++){
            if (!isEqual(dense_matrix[i][j], 0.0)){
                // store column index and value directly
                column_indices[nz_idx] = j;
                values[nz_idx] = dense_matrix[i][j];
                nz_idx++;
            }
        }
    }
    row_offsets[M] = nz;
}


void csr_to_file(const char* file, int M, int N, int nz, int* row_offsets, int* column_indices, float* values){
    ofstream csr_f;
    string file_str = (string) file;
    csr_f.open("test_matrices/csr/" + file_str.substr(14, file_str.size()-18) + "_csr.csv");

    csr_f << M << "\n";   // rows
    csr_f << N << "\n";   // columns
    csr_f << nz << "\n";  // number of non-zero elements

    // row offsets
    for (int i=0; i<M+1; i++){
        csr_f << row_offsets[i] << ", ";
    }
    csr_f << "\n";

    // columns indices
    for (int i=0; i<nz; i++){
        csr_f << column_indices[i] << ", ";
    }
    csr_f << "\n";

    // values
    for (int i=0; i<nz; i++){
        csr_f << values[i] << ", ";
    }
    csr_f << "\n";

    csr_f.close();
}

void transposed_csr_to_file(std::string file, int M, int N, int nz, int* row_offsets, int* column_indices, float* values){
    ofstream csr_f;
    csr_f.open("test_matrices/transposed/" + file.substr(18, file.size()));

    csr_f << M << "\n";   // rows
    csr_f << N << "\n";   // columns
    csr_f << nz << "\n";  // number of non-zero elements

    // row offsets
    for (int i=0; i<M+1; i++){
        csr_f << row_offsets[i] << ", ";
    }
    csr_f << "\n";

    // columns indices
    for (int i=0; i<nz; i++){
        csr_f << column_indices[i] << ", ";
    }
    csr_f << "\n";

    // values
    for (int i=0; i<nz; i++){
        csr_f << values[i] << ", ";
    }
    csr_f << "\n";

    csr_f.close();
}


void to_coo(float** dense_matrix, int M, int N, int nz, int* row_indices, int* column_indices, float* values){
    int nz_idx = 0;
    
    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            if (!isEqual(dense_matrix[i][j], 0.0)){
                // store row and column index and value
                row_indices[nz_idx] = i;
                column_indices[nz_idx] = j;
                values[nz_idx] = dense_matrix[i][j];
                nz_idx++;
            }
        }
    }
}

void coo_to_file(const char* file, int M, int N, int nz, int* row_indices, int* column_indices, float* values){
    ofstream coo_f;
    string file_str = (string) file;
    coo_f.open("test_matrices/coo/" + file_str.substr(14, file_str.size()-18) + "_coo.csv");

    coo_f << M << "\n";   // rows
    coo_f << N << "\n";   // columns
    coo_f << nz << "\n";  // number of non-zero elements

    // row offsets
    for (int i=0; i<nz; i++){
        coo_f << row_indices[i] << ", ";
    }
    coo_f << "\n";

    // columns indices
    for (int i=0; i<nz; i++){
        coo_f << column_indices[i] << ", ";
    }
    coo_f << "\n";

    // values
    for (int i=0; i<nz; i++){
        coo_f << values[i] << ", ";
    }
    coo_f << "\n";

    coo_f.close();
}


void convert_mtx_to_file(const char* file){
    // open file
    FILE *f;
    f = fopen(file, "r");

    // read matrix descriptor
    MM_typecode matcode;
    mm_read_banner(f, &matcode);
    
    // read size information and number of non-zero elements
    int M, N, nz; 
    mm_read_mtx_crd_size(f, &M, &N, &nz);

    // create dense matrix 
    float** dense_matrix = new float*[M];

    // fill with zeros
    for (int i=0; i<M; i++){
        dense_matrix[i] = new float[N];
        for (int j=0; j<N; j++){
            dense_matrix[i][j] = 0.0;
        }
    }

    // fill in the non-zero values
    int row, column;
    double value;

    for (int i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &row, &column, &value);
        row--;  /* adjust from 1-based to 0-based */
        column--;

        dense_matrix[row][column] = (float) value;
    }

    // convert to CSR
    int row_offsets[M+1];
    int column_indices[nz];
    float values[nz];

    to_csr(dense_matrix, M, N, nz, row_offsets, column_indices, values);   

    // save CSR to file
    csr_to_file(file, M, N, nz, row_offsets, column_indices, values);
    

    // convert to COO
    int row_indices_coo[nz];
    int column_indices_coo[nz];
    float values_coo[nz];

    to_coo(dense_matrix, M, N, nz, row_indices_coo, column_indices_coo, values_coo);

    // save COO to file
    coo_to_file(file, M, N, nz, row_indices_coo, column_indices_coo, values_coo);

}


void coo_from_file(string file, int &rows, int &cols, int &nnz, int*& row_indices, int*& col_indices, float*& values){
    ifstream coo_f(file);
    string line; 

    // first line is number of rows
    getline(coo_f, line);
    rows = stoi(line);

    // second line is number of cols
    getline(coo_f, line);
    cols = stoi(line);

    // third line is number of non-zero elements
    getline(coo_f, line);
    nnz = stoi(line);

    // fourth line are the row indices
    row_indices = (int*) malloc(nnz * sizeof(int));

    for (int i=0; i<nnz; i++){
        std::getline(coo_f, line, ',');  
        row_indices[i] = stoi(line);
    }

    // five line are the column indices
    col_indices = (int*) malloc(nnz * sizeof(int));

    for (int i=0; i<nnz; i++){
        std::getline(coo_f, line, ',');  
        col_indices[i] = stoi(line);
    }

    // sixth line are the values
    values = (float*) malloc(nnz * sizeof(float));

    for (int i=0; i<nnz; i++){
        std::getline(coo_f, line, ',');  
        values[i] = stoi(line);
    }

    coo_f.close();
}


void csr_from_file(string file, int &rows, int &cols, int &nnz, int*& row_offsets, int*& col_indices, float*& values){
    ifstream csr_f(file);
    string line; 

    // first line is number of rows
    getline(csr_f, line);
    rows = stoi(line);

    // second line is number of cols
    getline(csr_f, line);
    cols = stoi(line);

    // third line is number of non-zero elements
    getline(csr_f, line);
    nnz = stoi(line);

    // fourth line are the row offsets
    row_offsets = (int*) malloc((rows+1) * sizeof(int));

    for (int i=0; i<rows+1; i++){
        std::getline(csr_f, line, ',');  
        row_offsets[i] = stoi(line);
    }

    // five line are the column indices
    col_indices = (int*) malloc(nnz * sizeof(int));

    for (int i=0; i<nnz; i++){
        std::getline(csr_f, line, ',');  
        col_indices[i] = stoi(line);
    }

    // sixth line are the values
    values = (float*) malloc(nnz * sizeof(float));

    for (int i=0; i<nnz; i++){
        std::getline(csr_f, line, ',');  
        values[i] = stoi(line);
    }

    csr_f.close();
}
