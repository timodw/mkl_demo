#include <iostream>
#include <iomanip>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <map>
#include <vector>
#include <array>
#include <utility>
#include <boost/functional/hash.hpp>
#include <stdlib.h>
#include <math.h>

#include "main.h"

int main(int argc, char** argv) {
    int mat_size = atoi(argv[1]);
    Eigen::initParallel();
    std::cout << "THREADS: " << Eigen::nbThreads() << std::endl;

    // Generate matrices
    double* mat = generate_random_matrix(mat_size);
    CSR* csr = generate_random_csr(mat_size, 0.01);
    double* out = new double[mat_size * mat_size];

    // Run base benchmark
    bench_base_A_mul_Bn(out, csr, mat, mat_size);    

    // Run Eigen benchmark
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_mat =
                 Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(mat, mat_size, mat_size);
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_csr = csr_to_eigen(csr);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_out = bench_eigen_A_mul_Bn(eigen_csr, &eigen_mat);

    // See if calculations were correct
    std::cout << (compare_matrices(out, eigen_out.data(), mat_size) ? "CORRECT" : "WRONG") << std::endl;

    // Cleanup
    delete[] mat;
    delete[] out;
    delete eigen_csr;
    free_csr(csr);
    free(csr);
    return 0;
}

// Generates a random vector containing doubles of a given size
double* generate_random_vector(long size) {
    std::default_random_engine ge;
    std::uniform_real_distribution<double> distr(0.0, 1.0);
    double* vec = new double[size];
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < size; i++) {
        vec[i] = distr(ge);
    }
    return vec;
}

// Generates a random square matrix containting doubles of size N x N where N is equal to the size parameter
double* generate_random_matrix(long size) {
    return generate_random_vector(size * size);
}

// Generates a random square sparse matrix with a given sparsness
CSR* generate_random_csr(long size, double sparsness) {
    // Initialize variables
    long nnz = (long) size * size * sparsness;
    int* rows = new int[nnz];
    int* cols = new int[nnz];
    double* vals = new double[nnz];

    // Random number generators
    std::default_random_engine ge;
    std::uniform_real_distribution<double> double_distr(0.0, 1.0);
    std::uniform_int_distribution<long> int_distr(0, size - 1);

    // Set used to know which coordinates already have a value
    std::unordered_set<std::pair<long, long>, boost::hash<std::pair<long, long>>> coordinate_set;

    omp_lock_t setlock;
    omp_init_lock(&setlock);
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < nnz; i++) {
        vals[i] = double_distr(ge);

        // Get random row and column to store this value at
        long row = int_distr(ge);
        long col = int_distr(ge);
        std::pair<long, long> coordinate = std::make_pair(row, col);
        omp_set_lock(&setlock);
        bool in_set = coordinate_set.insert(coordinate).second; 
        omp_unset_lock(&setlock);
        while (!in_set) {
            row = int_distr(ge);
            col = int_distr(ge);
            coordinate = std::make_pair(row, col);
            omp_set_lock(&setlock);
            in_set = coordinate_set.insert(coordinate).second; 
            omp_unset_lock(&setlock);
        }
        rows[i] = row;
        cols[i] = col;
    }

    // Create CSR
    struct CSR* csr = (struct CSR*) malloc(sizeof(struct CSR));
    new_csr(csr, nnz, size, size, rows, cols, vals);

    return csr;
}

// Prints a square matrix given by a double array
void print_matrix(double* mat, long size) {
    std::cout << std::setw(5) << std::setprecision(2);
    for (size_t i = 0; i < size * size; i++) {
        if (i != 0 && i % size == 0) std::cout << "\n";
        std::cout << mat[i] << "\t";
    }
    std::cout << std::endl;
    std::cout << std::setprecision(6);
}

// Compares 2 square matrices given by a double array and returns if they are equal or not
// (rounded to 3 significant digits)
bool compare_matrices(double* comp_1, double* comp_2, long size) {
    bool correct = true;
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < size * size; i++) {
        if (round(comp_1[i] * 1000.0) / 1000.0 != round(comp_2[i] * 1000.0) / 1000.0) correct = false;
    }
    return correct;
}

// Times the calculation of a CSR times a square matrix using the base implementation from Smurff
void bench_base_A_mul_Bn(double* Y, struct CSR* A, double* X, const int ncol) {
    // Calculate base matrix product
    std::cout << "START BASE CALCULATION" << "\n";
    double stime = omp_get_wtime();
    base_A_mul_Bn(Y, A, X, ncol);
    double etime = omp_get_wtime();
    std::cout << "BASE DURATION: " << etime - stime << "\n" << std::endl;
}

// Times the calculation of an Eigen::SparseMatrix times an Eigen::Matrix
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bench_eigen_A_mul_Bn(
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* eigen_mat
) {
    std::cout << "START EIGEN CALCULATION" << "\n";
    double stime = omp_get_wtime();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_eigen = (*eigen_sparse) * (*eigen_mat);
    double etime = omp_get_wtime();
    std::cout << "EIGEN DURATION: " << etime - stime << "\n" << std::endl;
    return out_eigen;
}

// Converts a CSR from Smurff to an Eigenn::SparseMatrix
Eigen::SparseMatrix<double, Eigen::RowMajor>* csr_to_eigen(CSR* csr) {
    Eigen::SparseMatrix<double, Eigen::RowMajor>* sm = new Eigen::SparseMatrix<double, Eigen::RowMajor>(csr->nrow, csr->ncol);
    std::vector<Eigen::Triplet<double>>* triplets = new std::vector<Eigen::Triplet<double>>();

    long counter = 0;
    for (size_t i = 1; i <= csr->nrow; i++) {
        for (size_t j = csr->row_ptr[i - 1]; j < csr->row_ptr[i]; j++) {
            triplets->push_back(Eigen::Triplet<double>(i - 1, csr->cols[j], csr->vals[counter++]));
        }
    }

    sm->setFromTriplets(triplets->begin(), triplets->end());

    delete triplets;

    return sm;
}

 /** y = A * x */
inline void base_A_mul_B(double* y, struct CSR *A, double* x) 
{
   int* row_ptr = A->row_ptr;
   int* cols    = A->cols;
   double* vals = A->vals;
   #pragma omp parallel for schedule(dynamic, 256)
   for (int row = 0; row < A->nrow; row++) 
   {
     double tmp = 0;
     const int end = row_ptr[row + 1];
     for (int i = row_ptr[row]; i < end; i++) 
     {
       tmp += x[cols[i]] * vals[i];
     }
     y[row] = tmp;
   }
 }

 /** Y = A * X, where Y and X have <ncol> columns and are row-ordered */
inline void base_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol) 
{
   int* row_ptr = A->row_ptr;
   int* cols    = A->cols;
   double* vals = A->vals;
   #pragma omp parallel
   {
     double* tmp = (double*)malloc(ncol * sizeof(double));
     #pragma omp parallel for schedule(dynamic, 256)
     for (int row = 0; row < A->nrow; row++) {
       memset(tmp, 0, ncol * sizeof(double));
       for (int i = row_ptr[row], end = row_ptr[row + 1]; i < end; i++) 
       {
         int col = cols[i] * ncol;
         double val = vals[i];
         for (int j = 0; j < ncol; j++) 
         {
            tmp[j] += X[col + j] * val;
         }
       }
       int r = row * ncol;
       for (int j = 0; j < ncol; j++) 
       {
         Y[r + j] = tmp[j];
       }
     }
     free(tmp);
   }
 }