#pragma once

#include "LibFastSparseDependency.h"

namespace smurff {

class SparseFeat
{
public:
   BinaryCSR M;
   BinaryCSR Mt;

   SparseFeat() {}

   SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols)
   {
      new_bcsr(&M, nnz, nrow, ncol, rows, cols);
      new_bcsr(&Mt, nnz, ncol, nrow, cols, rows);
   }

   virtual ~SparseFeat()
   {
      free_bcsr(&M);
      free_bcsr(&Mt);
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
