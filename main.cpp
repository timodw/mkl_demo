#define EIGEN_HAS_OPENMP

#include "bench.h"

//#define BENCH

int main(int argc, char** argv) {
    int mat_size = atoi(argv[1]);
    Eigen::initParallel();
    run_A_mul_B_benches(mat_size);
    return 0;
}