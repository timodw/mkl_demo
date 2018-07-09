#pragma once

#ifdef _WINDOWS
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#include <memory>
#include <string.h>

#include "Error.h"

inline long read_long(FILE* fh) {
   long value;
   size_t result1 = fread(&value, sizeof(long), 1, fh);
   if (result1 != 1) {
     fprintf( stderr, "File reading error for a long. File is corrupt.\n");
     exit(1);
   }
   return value;
 }

/*** binary CSR ***/
struct BinaryCSR
{
  int nrow;
  int ncol;
  long nnz;
  int* row_ptr; /* points to the row starts for each row */
  int* cols;
};

inline void free_bcsr(struct BinaryCSR* bcsr) 
{
   THROWERROR_ASSERT(bcsr != 0);

   free(bcsr->row_ptr);
   free(bcsr->cols);
 }
 
 static inline void new_bcsr(struct BinaryCSR* RESTRICT A, long nnz, int nrow, int ncol, int* rows, int* cols) 
 {
   THROWERROR_ASSERT(A != 0);

   //struct BinaryCSR *A = (struct BinaryCSR*)malloc(sizeof(struct BinaryCSR));
   A->nnz  = nnz;
   A->nrow = nrow;
   A->ncol = ncol;
   A->cols = (int*)malloc(nnz * sizeof(int));
   A->row_ptr = (int*)malloc( (nrow + 1) * sizeof(int));
 
   //compute number of non-zero entries per row of A
   for (int row = 0; row < nrow; row++) {
     A->row_ptr[row] = 0;
   }
 
   for (int i = 0; i < nnz; i++) {
     A->row_ptr[rows[i]]++;
   }
   // cumsum counts
   for (int row = 0, cumsum = 0; row < nrow; row++) {
     int temp = A->row_ptr[row];
     A->row_ptr[row] = cumsum;
     cumsum += temp;
   }
   A->row_ptr[nrow] = nnz;
 
   // writing cols and vals to A->cols and A->vals
   for (int i = 0; i < nnz; i++) {
     int row  = rows[i];
     int dest = A->row_ptr[row];
     A->cols[dest] = cols[i];
     A->row_ptr[row]++;
   }
   for (int row = 0, prev = 0; row <= nrow; row++) {
     int temp        = A->row_ptr[row];
     A->row_ptr[row] = prev;
     prev            = temp;
   }
 }

/*** Double CSR ***/
struct CSR
{
  int nrow;
  int ncol;
  long nnz;
  int* row_ptr; /* points to the row starts for each row */
  int* cols;
  double* vals;
};

inline void free_csr(struct CSR* csr) 
{
   THROWERROR_ASSERT(csr != 0);

   free(csr->row_ptr);
   free(csr->cols);
   free(csr->vals);
 }
 
 static inline void new_csr(
     struct CSR* RESTRICT A,
     long nnz,
     int nrow,
     int ncol,
     int* rows,
     int* cols,
     double* vals)
 {
   THROWERROR_ASSERT(A != 0);

   A->nnz  = nnz;
   A->nrow = nrow;
   A->ncol = ncol;
   A->cols    = (int*)malloc(nnz * sizeof(int));
   A->vals    = (double*)malloc(nnz * sizeof(double));
   A->row_ptr = (int*)malloc( (nrow + 1) * sizeof(int));
 
   //compute number of non-zero entries per row of A
   for (int row = 0; row < nrow; row++) {
     A->row_ptr[row] = 0;
   }
 
   for (int i = 0; i < nnz; i++) {
     A->row_ptr[rows[i]]++;
   }
   // cumsum counts
   for (int row = 0, cumsum = 0; row < nrow; row++) {
     int temp = A->row_ptr[row];
     A->row_ptr[row] = cumsum;
     cumsum += temp;
   }
   A->row_ptr[nrow] = nnz;
 
   // writing cols and vals to A->cols and A->vals
   for (int i = 0; i < nnz; i++) {
     int row = rows[i];
     int dest = A->row_ptr[row];
     A->cols[dest] = cols[i];
     A->vals[dest] = vals[i];
 
     A->row_ptr[row]++;
   }
   for (int row = 0, prev = 0; row <= nrow; row++) {
     int temp        = A->row_ptr[row];
     A->row_ptr[row] = prev;
     prev            = temp;
   }
 }

 /** y = A * x */
inline void bcsr_A_mul_B(double* y, struct BinaryCSR *A, double* x) 
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

/** Y = A * X, where Y and X have <ncol> columns and are row-ordered */
inline void bcsr_A_mul_Bn(double* Y, struct BinaryCSR *A, double* X, const int ncol) 
{
   int* row_ptr = A->row_ptr;
   int* cols    = A->cols;
   #pragma omp parallel
   {
     double* tmp = (double*)malloc(ncol * sizeof(double));
     #pragma omp parallel for schedule(dynamic, 256)
     for (int row = 0; row < A->nrow; row++) 
     {
       memset(tmp, 0, ncol * sizeof(double));
       const int end = row_ptr[row + 1];
       for (int i = row_ptr[row]; i < end; i++) 
       {
         int col = cols[i] * ncol;
         for (int j = 0; j < ncol; j++) 
         {
            tmp[j] += X[col + j];
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

 /** y = A * x */
inline void csr_A_mul_B(double* y, struct CSR *A, double* x) 
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
inline void csr_A_mul_Bn(double* Y, struct CSR *A, double* X, const int ncol) 
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

struct SparseBinaryMatrix
{
  int nrow;
  int ncol;
  long nnz;
  int* rows;
  int* cols;
};

inline void free_sbm(struct SparseBinaryMatrix* sbm) {
   free( sbm->rows );
   free( sbm->cols );
 }

 /** constructor, computes nrow and ncol from data */
inline struct SparseBinaryMatrix* new_sbm(long nrow, long ncol, long nnz, int* rows, int* cols) {
   struct SparseBinaryMatrix *A = (struct SparseBinaryMatrix*)malloc(sizeof(struct SparseBinaryMatrix));
   A->nnz  = nnz;
   A->rows = rows;
   A->cols = cols;
   A->nrow = nrow;
   A->ncol = ncol;
   return A;
 }

 inline struct SparseBinaryMatrix* read_sbm(const char *filename) {
   FILE* fh = fopen( filename, "r" );
   size_t result1, result2;
   if (fh == NULL) {
     fprintf( stderr, "File error: %s\n", filename );
     exit(1);
   }
   long nrow = read_long(fh);
   long ncol = read_long(fh);
   long nnz  = read_long(fh);
   // reading data
   int* rows = (int*)malloc(nnz * sizeof(int));
   int* cols = (int*)malloc(nnz * sizeof(int));
   result1 = fread(rows, sizeof(int), nnz, fh);
   result2 = fread(cols, sizeof(int), nnz, fh);
   if ((int)result1 != nnz || (int)result2 != nnz) {
     fprintf( stderr, "File read error: %s\n", filename );
     exit(1);
   }
   fclose(fh);
   // convert data from 1 based to 0 based
   for (long i = 0; i < nnz; i++) {
     rows[i]--;
     cols[i]--;
   }
 
   return new_sbm(nrow, ncol, nnz, rows, cols);
 }

 struct SparseDoubleMatrix
 {
   int nrow;
   int ncol;
   long nnz;
   int* rows;
   int* cols;
   double* vals;
 };

 // function that was missing in libfastsparse
 
 inline void free_sdm(SparseDoubleMatrix* sdm)
 {
    free(sdm->rows);
    free(sdm->cols);
    free(sdm->vals);
 }

 /** constructor, computes nrow and ncol from data */
inline struct SparseDoubleMatrix* new_sdm(long nrow, long ncol, long nnz, int* rows, int* cols, double* vals) {
   struct SparseDoubleMatrix *A = (struct SparseDoubleMatrix*)malloc(sizeof(struct SparseDoubleMatrix));
   A->nnz  = nnz;
   A->rows = rows;
   A->cols = cols;
   A->vals = vals;
   A->nrow = nrow;
   A->ncol = ncol;
   return A;
 }

inline struct SparseDoubleMatrix* read_sdm(const char *filename) {
   FILE* fh = fopen( filename, "r" );
   size_t result1, result2, result3;
   if (fh == NULL) {
     fprintf( stderr, "File error: %s\n", filename );
     exit(1);
   }
   unsigned long nrow = read_long(fh);
   unsigned long ncol = read_long(fh);
   unsigned long nnz  = read_long(fh);
   // reading data
   int* rows = (int*)malloc(nnz * sizeof(int));
   int* cols = (int*)malloc(nnz * sizeof(int));
   double* vals = (double*)malloc(nnz * sizeof(double));
   result1 = fread(rows, sizeof(int), nnz, fh);
   result2 = fread(cols, sizeof(int), nnz, fh);
   result3 = fread(vals, sizeof(double), nnz, fh);
   if (result1 != nnz || result2 != nnz || result3 != nnz) {
     fprintf( stderr, "File read error: %s\n", filename );
     exit(1);
   }
   fclose(fh);
   // convert data from 1 based to 0 based
   for (unsigned long i = 0; i < nnz; i++) {
     rows[i]--;
     cols[i]--;
   }
 
   return new_sdm(nrow, ncol, nnz, rows, cols, vals);
 } 