// main.cpp
// Parallel Matrix Computation Pipeline using Taskflow
//
// Tasks:
//  A: Generate random matrices (A_mat, B_mat)
//  B: Blocked matrix multiplication of A_mat * B_mat -> M_prod  (uses subflow tasks per tile)
//  C: Element-wise addition A_mat + B_mat -> M_sum (uses subflow tasks per tile)
//  D: Verification, timing summary, write DOT graph
//
// Build: see CMakeLists.txt

#include <taskflow/taskflow.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

using Matrix = std::vector<double>;

// Helper: index into row-major matrix
inline size_t idx(size_t r, size_t c, size_t N) { return r * N + c; }

// Fill matrix with random values
void fill_random(Matrix &M, size_t N, unsigned seed=1234) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(size_t i=0;i<N*N;++i) M[i] = dist(rng);
}

// Sequential naive multiplication (for baseline)
void matmul_naive(const Matrix &A, const Matrix &B, Matrix &C, size_t N) {
    std::fill(C.begin(), C.end(), 0.0);
    for(size_t i=0;i<N;++i) {
        for(size_t k=0;k<N;++k) {
            double aik = A[idx(i,k,N)];
            for(size_t j=0;j<N;++j) {
                C[idx(i,j,N)] += aik * B[idx(k,j,N)];
            }
        }
    }
}

// Blocked multiplication for a single block (i0..i1-1, j0..j1-1), uses full k
void matmul_block(const Matrix &A, const Matrix &B, Matrix &C,
                  size_t N, size_t i0, size_t i1, size_t j0, size_t j1, size_t block_k = 0, size_t k0 = 0, size_t k1 = 0)
{
    // Here we compute over k in [0,N) (classical) unless k0,k1 provided.
    if(k1 == 0) { k0 = 0; k1 = N; }
    for(size_t i = i0; i < i1; ++i) {
        for(size_t k = k0; k < k1; ++k) {
            double aik = A[idx(i,k,N)];
            size_t base = i*N;
            size_t rowk = k*N;
            for(size_t j = j0; j < j1; ++j) {
                C[base + j] += aik * B[rowk + j];
            }
        }
    }
}

// Element-wise add block: C = A + B for rows i0..i1-1, cols j0..j1-1
void add_block(const Matrix &A, const Matrix &B, Matrix &C, size_t N, size_t i0, size_t i1, size_t j0, size_t j1) {
    for(size_t i = i0; i < i1; ++i) {
        for(size_t j = j0; j < j1; ++j) {
            C[idx(i,j,N)] = A[idx(i,j,N)] + B[idx(i,j,N)];
        }
    }
}

// Verify that two matrices are close within tolerance
bool verify_close(const Matrix &X, const Matrix &Y, size_t N, double tol = 1e-8) {
    for(size_t i=0;i<N*N;++i) {
        double a = X[i], b = Y[i];
        if(std::isnan(a) || std::isnan(b)) return false;
        if(std::abs(a-b) > tol * std::max(1.0, std::abs(b))) return false;
    }
    return true;
}

int main(int argc, char** argv) {

    // Configurable parameters (matrix size N, block size tile)
    size_t N = 512;          // matrix dimension (NxN)
    size_t tile = 64;        // tile size (both dimensions)
    int seed = 1234;
    bool dump_dot = true;

    // Parse simple args
    for(int i=1;i<argc;i++){
        std::string s = argv[i];
        if(s == "-n" && i+1<argc) N = static_cast<size_t>(std::stoul(argv[++i]));
        else if(s == "-t" && i+1<argc) tile = static_cast<size_t>(std::stoul(argv[++i]));
        else if(s == "-s" && i+1<argc) seed = std::stoi(argv[++i]);
        else if(s == "--no-dot") dump_dot = false;
        else if(s == "-h" || s == "--help") {
            std::cout << "Usage: " << argv[0] << " [-n N] [-t tile] [-s seed] [--no-dot]\n";
            return 0;
        }
    }

    std::cout << "Matrix N="<<N<<", tile="<<tile<<"\n";

    // Allocate matrices
    Matrix A_mat(N*N), B_mat(N*N);
    Matrix M_prod_parallel(N*N); // product computed by Taskflow blocked
    Matrix M_prod_seq(N*N);      // product by naive sequential
    Matrix M_sum(N*N);           // A + B computed by Taskflow

    // For timing
    double t_gen=0, t_seq=0, t_parallel=0, t_add=0, t_verify=0;

    // Taskflow setup
    tf::Executor executor;
    tf::Taskflow taskflow("MatrixPipeline");

    // A: generate matrices (this runs before B and C)
    auto taskA = taskflow.emplace([&](){
        auto t0 = Clock::now();
        fill_random(A_mat, N, seed);
        fill_random(B_mat, N, seed+1);
        auto t1 = Clock::now();
        t_gen = ms(t1 - t0).count();
        std::cout << "[A] Generated matrices in " << t_gen << " ms\n";
    }).name("A:generate");

    // B: blocked parallel multiplication using subflow
    auto taskB = taskflow.emplace([&](tf::Subflow& sf){
        // initialize product matrix
        std::fill(M_prod_parallel.begin(), M_prod_parallel.end(), 0.0);

        size_t tiles = (N + tile - 1) / tile;
        auto tb0 = Clock::now();

        // create a task per (i_tile, j_tile) that computes the block result by iterating k over full range
        // Option: we could create tasks per (i_tile, j_tile, k_tile) and reduce; here do i-j tasks with full k loop.
        for(size_t it = 0; it < tiles; ++it) {
            size_t i0 = it * tile;
            size_t i1 = std::min(N, i0 + tile);
            for(size_t jt = 0; jt < tiles; ++jt) {
                size_t j0 = jt * tile;
                size_t j1 = std::min(N, j0 + tile);

                // each subtask computes C[i0:i1, j0:j1] contribution by iterating k
                sf.emplace([=, &A_mat, &B_mat, &M_prod_parallel, N]() {
                    matmul_block(A_mat, B_mat, M_prod_parallel, N, i0, i1, j0, j1);
                });
            }
        }

        // join subflow before finishing taskB
        sf.join();

        auto tb1 = Clock::now();
        t_parallel = ms(tb1 - tb0).count();
        std::cout << "[B] Blocked parallel multiplication finished in " << t_parallel << " ms\n";
    }).name("B:block-matmul");

    // C: compute element-wise sum A + B in parallel using subflow
    auto taskC = taskflow.emplace([&](tf::Subflow& sf){
        std::fill(M_sum.begin(), M_sum.end(), 0.0);
        size_t tiles = (N + tile - 1) / tile;
        auto t0 = Clock::now();
        for(size_t it = 0; it < tiles; ++it) {
            size_t i0 = it * tile;
            size_t i1 = std::min(N, i0 + tile);
            for(size_t jt = 0; jt < tiles; ++jt) {
                size_t j0 = jt * tile;
                size_t j1 = std::min(N, j0 + tile);
                sf.emplace([=, &A_mat, &B_mat, &M_sum, N]() {
                    add_block(A_mat, B_mat, M_sum, N, i0, i1, j0, j1);
                });
            }
        }
        sf.join();
        auto t1 = Clock::now();
        t_add = ms(t1 - t0).count();
        std::cout << "[C] Blocked add finished in " << t_add << " ms\n";
    }).name("C:block-add");

    // D: verification and sequential baseline
    auto taskD = taskflow.emplace([&](){
        auto t_verify_start = Clock::now();

        // Sequential baseline (naive)
        auto t0 = Clock::now();
        matmul_naive(A_mat, B_mat, M_prod_seq, N);
        auto t1 = Clock::now();
        t_seq = ms(t1 - t0).count();

        // Verify parallel product matches sequential
        bool prod_ok = verify_close(M_prod_seq, M_prod_parallel, N, 1e-6);
        bool add_ok = verify_close(M_sum, [&]() -> Matrix {
            Matrix tmp(N*N);
            for(size_t i=0;i<N*N;++i) tmp[i] = A_mat[i] + B_mat[i];
            return tmp;
        }(), N, 1e-6);

        auto t_verify_end = Clock::now();
        t_verify = ms(t_verify_end - t_verify_start).count();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n==== Timing Summary (ms) ====\n";
        std::cout << "Generation: \t" << t_gen << "\n";
        std::cout << "Sequential mult (naive): \t" << t_seq << "\n";
        std::cout << "Parallel blocked mult (Taskflow subflow): \t" << t_parallel << "\n";
        std::cout << "Blocked add (Taskflow subflow): \t" << t_add << "\n";
        std::cout << "Verification time: \t" << t_verify << "\n";
        double speedup = t_seq / (t_parallel + 1e-9);
        std::cout << "Speedup (seq / parallel): \t" << speedup << "x\n";

        std::cout << "\nVerification results:\n";
        std::cout << "Product match: " << (prod_ok? "OK": "MISMATCH") << "\n";
        std::cout << "Add match:     " << (add_ok? "OK": "MISMATCH") << "\n";

        if(!prod_ok || !add_ok) {
            std::cerr << "ERROR: results mismatch!\n";
        }

        // Optionally dump DOT graph (Taskflow can dump via to_dot)
        if(dump_dot) {
            std::ofstream ofs("taskflow_graph.dot");
            if(ofs) {
                taskflow.dump(ofs); // Taskflow provides dump method to output dot format
                ofs.close();
                std::cout << "Wrote Taskflow graph to taskflow_graph.dot\n";
            } else {
                std::cerr << "Failed to open taskflow_graph.dot for writing.\n";
            }
        }
    }).name("D:verify");

    // Dependencies: A -> {B, C} -> D
    taskA.precede(taskB);
    taskA.precede(taskC);
    taskB.precede(taskD);
    taskC.precede(taskD);

    // Run the flow
    executor.run(taskflow).wait();

    return 0;
}
