#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "LibFastSparseDependency.h"

typedef struct {
    int nrow;
    int ncol;
    double* values;
    int* cols;
    int* row_begins;
    int* row_ends;
} mkl_csr;

void base_A_mul_B(double* y, struct CSR *A, double* x);
void base_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol);

void mkl_A_mul_B(double* y, mkl_csr A, double* x);
void mkl_A_mul_Bn(double* Y, mkl_csr A, double* X, const int ncol);

Eigen::SparseMatrix<double, Eigen::RowMajor>* csr_to_eigen(CSR*);
mkl_csr csr_to_mkl(CSR*); 

void bcsr_to_csr(CSR* csr, BinaryCSR* bcsr);