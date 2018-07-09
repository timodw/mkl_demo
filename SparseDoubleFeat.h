#pragma once

#include "LibFastSparseDependency.h"

namespace smurff {

class SparseDoubleFeat
{
public:
   CSR M;
   CSR Mt;

   SparseDoubleFeat() {}

   SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals)
   {
      new_csr(&M, nnz, nrow, ncol, rows, cols, vals);
      new_csr(&Mt, nnz, ncol, nrow, cols, rows, vals);
   }

   virtual ~SparseDoubleFeat()
   {
      free_csr(&M);
      free_csr(&Mt);
   }

   int nfeat() const
   {
      return M.ncol;
   }

   int cols() const
   {
      return M.ncol;
   }

   int nsamples() const
   {
      return M.nrow;
   }

   int rows() const
   {
      return M.nrow;
   }

   int nnz() const
   {
      return M.nnz;
   }
};

}
