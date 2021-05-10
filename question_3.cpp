#include <stdlib.h> 
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

typedef std::vector<int> vi;
typedef vector<vector<int>> matrix;

const int N = 30000; 
const int M = (N*N)/10; 


void naive_matrix_multiplication(const matrix& M, const vi& V){
    int output[N]={0};
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            output[i] += M[j][i]*V[i];
}

void CSR_matrix_multiplication(const matrix& M, const vi& V){
    // Convert to CSR representation
    vi A; // nonzero values in matrix
    vi IA = {0}; // row pointers
    vi JA; // column indices
    int NNZ = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            if (M[i][j]!=0) {
                A.push_back(M[i][j]);
                JA.push_back(j);
                NNZ++; // nonzeroes in row
            }
        }
        IA.push_back(NNZ);
    }
    // Calculate the result
    int output[N]={0};
    for(int i=0; i<N; i++)
        for(int j=IA[i]; j<IA[i+1]; j++)
            output[i] += A[j]*V[JA[j]];
}

void COO_matrix_multiplication(const matrix& M, const vi& V){
    // Convert to CSR representation
    vi A; // nonzero values in matrix
    vi R; // row indices
    vi C; // column indices
    int NNZ = 0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            if (M[i][j]!=0) {
                A.push_back(M[i][j]);
                R.push_back(i);
                C.push_back(j);
                NNZ++;
            }
        }
    }
    // Calculate the result
    int output[N]={0};
    for(int i=0; i<NNZ; i++)
        output[R[i]]+=A[i]*C[i];
}

int main(){
    // Initialize vector
    vi V(N, 0);
    for(int i=0; i<N; i++)
        V[i] = rand()%10+1;
    // Initialize massive array with 0s
    matrix sparse_matrix(N, vi(N)); // 2090916 elements in matrix
    // Sparsity
    for(int i=0; i<1000; i++)
        sparse_matrix[rand()%N][rand()%N] = rand()%10+1;
    // Tell how many nonzero elements were made.
    int total_elements = 0;
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
            if(sparse_matrix[i][j]>0)
                total_elements++;
    cout << "The matrix has a sparsity score of: " << (1-(double(total_elements))/(N*N))*100 << "%" << endl;
    // Find baseline naive performance
    cout << "Starting naive matrix-vector multiplication..." << endl;
    auto start = high_resolution_clock::now();
    naive_matrix_multiplication(sparse_matrix, V);
    auto end = high_resolution_clock::now();
    duration<double, std::milli> t = end-start;
    cout << "Naive matrix-vector finished in " << t.count() << " ms." << endl;
    // CSR representation
    cout << "Starting CSR matrix-vector multiplication..." << endl;
    start = high_resolution_clock::now();
    CSR_matrix_multiplication(sparse_matrix, V);
    end = high_resolution_clock::now();
    t = end-start;
    cout << "CSR matrix-vector finished in " << t.count() << " ms." << endl;
    // COO representation
    cout << "Starting COO matrix-vector multiplication..." << endl;
    start = high_resolution_clock::now();
    COO_matrix_multiplication(sparse_matrix, V);
    end = high_resolution_clock::now();
    t = end-start;
    cout << "COO matrix-vector finished in " << t.count() << " ms." << endl;
    return 0;
}