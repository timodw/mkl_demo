#include "bench.h"
#include <iostream>
#include <random>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <fstream>
#include <iomanip>

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

CSR* read_csr_from_file(std::string path) {
    std::ifstream fh(path);
    std::string input;
    getline(fh, input);
    getline(fh, input);
    getline(fh, input);

    // Read nrows, ncols and nnz
    int start = 0;
    int end = input.find(" ");
    int nrows = std::stoi(input.substr(start, end - start));
    start = end + 1;
    end = input.find(" ", start);
    int ncols = std::stoi(input.substr(start, end - start));
    int nnz = std::stoi(input.substr(end + 1));

    int* rows = new int[nnz];
    int* cols = new int[nnz];
    double* vals = new double[nnz];
    for (size_t i = 0; i < nnz; i++) {
        getline(fh, input);
        start = 0;
        end = input.find(" ");
        int row = std::stoi(input.substr(start, end - start)) - 1;
        start = end + 1;
        end = input.find(" ", start);
        int col = std::stoi(input.substr(start, end - start)) - 1;
        double val = (double) std::stoi(input.substr(end + 1));
        rows[i] = row;
        cols[i] = col;
        vals[i] = val;
    }

    CSR* csr = (CSR*) malloc(sizeof(CSR));
    new_csr(csr, nnz, nrows, ncols, rows, cols, vals);

    delete[] rows;
    delete[] cols;
    delete[] vals;
    return csr;
}

BinaryCSR* read_bcsr_from_file(std::string path) {
    std::ifstream fh(path);
    std::string input;
    getline(fh, input);
    getline(fh, input);
    getline(fh, input);

    // Read nrows, ncols and nnz
    int start = 0;
    int end = input.find(" ");
    int nrows = std::stoi(input.substr(start, end - start));
    start = end + 1;
    end = input.find(" ", start);
    int ncols = std::stoi(input.substr(start, end - start));
    int nnz = std::stoi(input.substr(end + 1));

    int* rows = new int[nnz];
    int* cols = new int[nnz];
    for (size_t i = 0; i < nnz; i++) {
        getline(fh, input);
        start = 0;
        end = input.find(" ");
        int row = std::stoi(input.substr(start, end - start)) - 1;
        int col = std::stoi(input.substr(end + 1)) - 1;
        rows[i] = row;
        cols[i] = col;
    }

    BinaryCSR* bcsr = (BinaryCSR*) malloc(sizeof(BinaryCSR));
    new_bcsr(bcsr, nnz, nrows, ncols, rows, cols);

    delete[] rows;
    delete[] cols;
    return bcsr;
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

// Compares 2 double arrays and returns if they are equal or not
// (rounded to 3 significant digits)
bool compare_vectors(double* comp_1, double* comp_2, long size) {
    bool correct = true;
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < size; i++) {
        if (round(comp_1[i] * 1000.0) / 1000.0 != round(comp_2[i] * 1000.0) / 1000.0) correct = false;
    }
    return correct;
}

void run_A_mul_B_benches(long size) {
    #ifndef BENCH
        std::cout << "THREADS: " << Eigen::nbThreads() << std::endl;
    #else
        std::cout << Eigen::nbThreads() << ";" << mat_size << ";";
    #endif

    BinaryCSR* bcsr = read_bcsr_from_file("/data/excape/excape_v4/side_info/ecfp_binary_full.mtx");
    CSR* csr = (CSR*) malloc(sizeof(CSR));
    bcsr_to_csr(csr, bcsr);
    double* vec = generate_random_vector(csr->ncol);
    double* out = new double[csr->nrow];

    bench_base_A_mul_B(out, bcsr, vec);

    Eigen::VectorXd eigen_vec = Eigen::Map<Eigen::VectorXd>(vec, csr->ncol);
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_csr = csr_to_eigen(csr);

    Eigen::VectorXd eigen_out = bench_eigen_A_mul_B(eigen_csr, &eigen_vec);

    mkl_csr mkl_sparse = csr_to_mkl(csr);
    bench_mkl_A_mul_B(out, mkl_sparse, vec);

    #ifndef BENCH
        std::cout << (compare_vectors(out, eigen_out.data(), csr->nrow) ? "CORRECT" : "WRONG") << std::endl;
    #endif

    delete[] vec;
    delete[] out;
    delete eigen_csr;
    free_csr(csr);
    free_bcsr(bcsr);
    free(csr);
    free(bcsr);
}

void run_A_mul_Bn_benches(long mat_size) {
    #ifndef BENCH
        std::cout << "THREADS: " << Eigen::nbThreads() << std::endl;
    #else
        std::cout << Eigen::nbThreads() << ";" << mat_size << ";";
    #endif

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

    // Run MKL benchmark
    mkl_csr sparse_mkl = csr_to_mkl(csr);
    bench_mkl_A_mul_Bn(out, sparse_mkl, mat, mat_size);

    // See if calculations were correct
    #ifndef BENCH
        std::cout << (compare_vectors(out, eigen_out.data(), mat_size * mat_size) ? "CORRECT" : "WRONG") << std::endl;
    #endif

    // Cleanup
    delete[] mat;
    delete[] out;
    delete eigen_csr;
    free_csr(csr);
    free(csr);
}

void bench_base_A_mul_B(double* y, struct BinaryCSR* A, double* x) {
    #ifndef BENCH
        std::cout << "START BASE CALCULATION" << "\n";
    #endif

    double total_time = 0;
    for (size_t i = 0; i < NR_ITERATIONS; i++) {
        double stime = omp_get_wtime();
        base_bcsr_A_mul_B(y, A, x);
        double etime = omp_get_wtime();
        total_time += etime - stime;
    }

    #ifndef BENCH
        std::cout << "BASE DURATION: " << total_time << "\n" << std::endl;
    #else
        std::cout << total_time << ";";
    #endif
}

Eigen::VectorXd bench_eigen_A_mul_B(Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse, Eigen::VectorXd* eigen_vec) {
    #ifndef BENCH
        std::cout << "START EIGEN CALCULATION" << "\n";
    #endif

    Eigen::VectorXd out_eigen;
    double total_time = 0;
    for (size_t i = 0; i < NR_ITERATIONS; i++) {
        double stime = omp_get_wtime();
        Eigen::VectorXd out_eigen = (*eigen_sparse) * (*eigen_vec);
        double etime = omp_get_wtime();
        total_time += etime - stime;
    }

    #ifndef BENCH
        std::cout << "EIGEN DURATION: " << total_time << "\n" << std::endl;
    #else
            std::cout << total_time << ";";
    #endif

    return out_eigen;
}

void bench_mkl_A_mul_B(double* y, mkl_csr A, double* x) {
    #ifndef BENCH
        std::cout << "START MKL CALCULATION" << "\n";
    #endif

    double total_time = 0;
    for (size_t i = 0; i < NR_ITERATIONS; i++) {
        double stime = omp_get_wtime();
        mkl_A_mul_B(y, A, x);
        double etime = omp_get_wtime();
        total_time += etime - stime;
    }

   #ifndef BENCH
        std::cout << "MKL DURATION: " << total_time << "\n" << std::endl;
    #else
            std::cout << total_time << ";";
    #endif 
}

// Times the calculation of a CSR times a square matrix using the base implementation from Smurff
void bench_base_A_mul_Bn(double* Y, struct CSR* A, double* X, const int ncol) {
    // Calculate base matrix product
    #ifndef BENCH
        std::cout << "START BASE CALCULATION" << "\n";
    #endif

    double stime = omp_get_wtime();
    base_A_mul_Bn(Y, A, X, ncol);
    double etime = omp_get_wtime();

    #ifndef BENCH
        std::cout << "BASE DURATION: " << etime - stime << "\n" << std::endl;
    #else
        std::cout << etime - stime << ";";
    #endif
}

// Times the calculation of an Eigen::SparseMatrix times an Eigen::Matrix
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bench_eigen_A_mul_Bn(
    Eigen::SparseMatrix<double, Eigen::RowMajor>* eigen_sparse,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* eigen_mat
) {
    #ifndef BENCH
        std::cout << "START EIGEN CALCULATION" << "\n";
    #endif

    double stime = omp_get_wtime();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_eigen = (*eigen_sparse) * (*eigen_mat);
    double etime = omp_get_wtime();

    #ifndef BENCH
        std::cout << "EIGEN DURATION: " << etime - stime << "\n" << std::endl;
    #else
            std::cout << etime - stime << ";";
    #endif

    return out_eigen;
}

void bench_mkl_A_mul_Bn(double* Y, mkl_csr A, double* X, const int ncol) {

    #ifndef BENCH
        std::cout << "START MKL CALCULATION" << "\n";
    #endif

    double stime = omp_get_wtime();
    mkl_A_mul_Bn(Y, A, X, ncol);
    double etime = omp_get_wtime();

    #ifndef BENCH
        std::cout << "MKL DURATION: " << etime - stime << "\n" << std::endl;
    #else
        std::cout << etime - stime << std::endl;
    #endif
}
