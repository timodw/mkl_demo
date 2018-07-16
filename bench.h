#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include "LibFastSparseDependency.h"
#include "matrix_operations.h"

#define NR_ITERATIONS 1

double* generate_random_vector(long size);
double* generate_random_matrix(long size);
CSR* generate_random_csr(long size, double sparsness);
CSR* read_csr_from_file(std::string path);
BinaryCSR* read_bcsr_from_file(std::string path);

void print_csr(CSR* csr);
void print_matrix(double* mat, long size);
bool compare_vectors(double* comp_1, double* comp_2, long size);

void run_A_mul_B_benches(long size);
void run_A_mul_Bn_benches(long size);

void bench_base_A_mul_B(double* y, struct BinaryCSR* A, double* x);
Eigen::VectorXd bench_eigen_A_mul_B(Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse, Eigen::VectorXd* eigen_vec);
void bench_mkl_A_mul_B(double* y, mkl_csr A, double* x);

void bench_base_A_mul_Bn(double* Y, struct CSR* A, double* X, const int ncol);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bench_eigen_A_mul_Bn(
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* eigen_mat
);
void bench_mkl_A_mul_Bn(double* Y, mkl_csr A, double* X, const int ncol);