#include "matrix_operations.h"
#include <mkl.h>

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

// Converts a Smurff CSR to a CSR struct that can be used by MKL
mkl_csr csr_to_mkl(CSR* csr) {
    mkl_csr sparse_mkl;
    sparse_mkl.nrow = csr->nrow;
    sparse_mkl.ncol = csr->ncol;
    sparse_mkl.values = csr->vals;
    sparse_mkl.cols = csr->cols;
    sparse_mkl.row_begins = csr->row_ptr;
    sparse_mkl.row_ends = (int*) malloc(sparse_mkl.nrow * sizeof(int));
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < sparse_mkl.nrow; i++) {
        sparse_mkl.row_ends[i] = sparse_mkl.row_begins[i + 1];
    }
    return sparse_mkl;
}

void bcsr_to_csr(CSR* csr, BinaryCSR* bcsr) {
    csr->nrow = bcsr->nrow;
    csr->ncol = bcsr->ncol;
    csr->nnz = bcsr->nnz;
    csr->row_ptr = (int*) malloc((bcsr->nrow + 1) * sizeof(int));
    csr->cols = (int*) malloc(bcsr->nnz * sizeof(int));
    csr->vals = (double*) malloc(bcsr->nnz * sizeof(double));

    memcpy(csr->row_ptr, bcsr->row_ptr, (bcsr->nrow + 1) * sizeof(int));
    memcpy(csr->cols, bcsr->cols, bcsr->nnz * sizeof(int));
    for (size_t i = 0; i < bcsr->nnz; i++) {
        csr->vals[i] = 1.0;
    }
}

 /** y = A * x */
void base_A_mul_B(double* y, struct CSR *A, double* x) {
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
void base_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol) {
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

void base_bcsr_A_mul_B(double* y, struct BinaryCSR *A, double* x) 
{
   int* row_ptr = A->row_ptr;
   int* cols    = A->cols;
   #pragma omp parallel for schedule(dynamic, 256)
   for (int row = 0; row < A->nrow; row++) {
     double tmp = 0;
     const int end = row_ptr[row + 1];
     for (int i = row_ptr[row]; i < end; i++) {
       tmp += x[cols[i]];
     }
     y[row] = tmp;
   }
 }


void mkl_A_mul_B(double* y, mkl_csr A, double* x) {
    char transa = 'N';
    int m = A.nrow;
    int k = A.ncol;
    double alpha = 1.0;
    char matdescra[] = {'G', ' ', ' ', 'C'};
    double beta = 0.0;

    mkl_dcsrmv(
        &transa,
        &m, &k,
        &alpha,
        matdescra,
        A.values,
        A.cols,
        A.row_begins,
        A.row_ends,
        x,
        &beta,
        y
    );
 }
 
void mkl_A_mul_Bn(double* Y, mkl_csr A, double* X, const int ncol) {
    char transa = 'N';
    int m = A.nrow;
    int n = ncol;
    int k = A.ncol;
    double alpha = 1.0;
    char matdescra[] = {'G', ' ', ' ', 'C'};
    double beta = 0.0;

    mkl_dcsrmm(
        &transa,
        &m, &n, &k,
        &alpha,
        matdescra,
        A.values,
        A.cols,
        A.row_begins,
        A.row_ends,
        X,
        &ncol,
        &beta,
        Y,
        &ncol
    );
}
 