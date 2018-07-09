#define EIGEN_HAS_OPENMP
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "LibFastSparseDependency.h"
#include "SparseFeat.h"
#include "SparseDoubleFeat.h"

// At * A
void At_mul_A(Eigen::MatrixXd & out, smurff::SparseFeat & A);
void At_mul_A(Eigen::MatrixXd & out, smurff::SparseDoubleFeat & A);

// AtA * B
void AtA_mul_B(Eigen::MatrixXd & out, smurff::SparseFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp);
void AtA_mul_B(Eigen::MatrixXd & out, smurff::SparseDoubleFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp);

// A * Bt
void A_mul_Bt( Eigen::MatrixXd & out, BinaryCSR & csr, Eigen::MatrixXd & B);
void A_mul_Bt( Eigen::MatrixXd & out, CSR & csr, Eigen::MatrixXd & B);

// A * b
void A_mul_B(  Eigen::VectorXd & out, BinaryCSR & csr, Eigen::VectorXd & b);
void A_mul_B(  Eigen::VectorXd & out, CSR & csr, Eigen::VectorXd & b);

// A * B
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, smurff::SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, smurff::SparseDoubleFeat & B);

void At_mul_Bt(Eigen::VectorXd & Y, smurff::SparseFeat & X, const int col, Eigen::MatrixXd & B);
void At_mul_Bt(Eigen::VectorXd & Y, smurff::SparseDoubleFeat & X, const int col, Eigen::MatrixXd & B);

void base_A_mul_B(double* y, struct CSR *A, double* x);
void base_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol);

double* generate_random_vector(long size);
double* generate_random_matrix(long size);
CSR* generate_random_csr(long size, double sparsness);

void print_csr(CSR* csr);
void print_matrix(double* mat, long size);
bool compare_matrices(double* mat_1, double* mat_2, long size);
void bench_base_A_mul_Bn(double* Y, struct CSR* A, double* X, const int ncol);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bench_eigen_A_mul_Bn(
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* eigen_mat
);

Eigen::SparseMatrix<double, Eigen::RowMajor>* csr_to_eigen(CSR*); 
