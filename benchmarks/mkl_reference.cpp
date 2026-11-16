/**
 * @file mkl_reference.cpp
 * @brief Measures MKL Pardiso SPD analysis, factorization and solve separately.
 *
 * Uses the same Matrix Market importer and manufactured solution as mfsolve.
 * Each trial performs a fresh numeric factorization; phase 11 is measured once.
 * Link this optional benchmark against MKL, matrix.cpp and memory.cpp.
 */
#include "matrix.h"
#include "load_matrix.h"
#include <mkl.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: mkl_reference matrix.mtx threads repeat [order.bin]\n";
            return 2;
        }
        const int threads = std::stoi(argv[2]);
        const int repeat = std::stoi(argv[3]);
        if (threads < 1 || repeat < 1) { throw std::runtime_error("Positive threads/repeat required"); }
        mkl_set_dynamic(0);
        mkl_set_num_threads(threads);
        const auto matrix = load_benchmark_matrix(argv[1]);
        MKL_INT n = matrix.n;
        std::vector<MKL_INT> ptr(matrix.ptr.begin(), matrix.ptr.end());
        std::vector<MKL_INT> row(matrix.row.begin(), matrix.row.end());
        std::vector<double> exact(n);
        for (int i = 0; i < n; ++i) {
            exact[i] = std::sin(0.17 * i) + 0.5 * std::cos(0.031 * i);
        }
        auto rhs = matrix.multiply(exact);
        std::vector<double> solution(n);

        // Lower CSC is upper CSR for this symmetric SPD matrix. Pardiso uses
        // its own default fill-reducing ordering and zero-based array indices.
        void* state[64]{};
        MKL_INT parameters[64]{};
        MKL_INT type = 2, max_factors = 1, factor = 1, nrhs = 1, messages = 0, error = 0;
        pardisoinit(state, &type, parameters);
        parameters[34] = 1;
        parameters[4] = argc == 5 ? 2 : 0;
        std::vector<MKL_INT> permutation(n);
        auto phase = [&](MKL_INT operation) {
            const auto start = std::chrono::steady_clock::now();
            pardiso(state, &max_factors, &factor, &type, &operation, &n,
                    matrix.value.data(), ptr.data(), row.data(), permutation.data(),
                    &nrhs, parameters, &messages, rhs.data(), solution.data(), &error);
            if (error) { throw std::runtime_error("Pardiso error " + std::to_string(error)); }
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        std::cout << std::unitbuf << std::setprecision(9);
        const double analysis = phase(11);
        if (argc == 5) {
            static_assert(sizeof(MKL_INT) == sizeof(int), "Order export requires LP64 MKL");
            std::ofstream output(argv[4], std::ios::binary);
            output.write(reinterpret_cast<const char*>(permutation.data()), n * sizeof(MKL_INT));
            output.close();
            if (!output) { throw std::runtime_error("Failed to write ordering"); }
        }
        std::cout << "solver=mkl_pardiso_spd matrix_n=" << n << " threads=" << threads
                  << " analysis_s=" << analysis << " factor_nnz=" << parameters[17] << '\n';
        for (int trial = 0; trial < repeat; ++trial) {
            const double factor_seconds = phase(22);
            const double solve_seconds = phase(33);
            const auto product = matrix.multiply(solution);
            double residual = 0, norm = 0, forward = 0, exact_norm = 0;
            for (int i = 0; i < n; ++i) {
                residual += (product[i] - rhs[i]) * (product[i] - rhs[i]);
                norm += rhs[i] * rhs[i];
                forward += (solution[i] - exact[i]) * (solution[i] - exact[i]);
                exact_norm += exact[i] * exact[i];
            }
            std::cout << "trial=" << trial + 1 << " factor_s=" << factor_seconds
                      << " solve_s=" << solve_seconds
                      << " numeric_total_s=" << factor_seconds + solve_seconds
                      << " relative_residual=" << std::sqrt(residual / norm)
                      << " relative_solution_error=" << std::sqrt(forward / exact_norm) << '\n';
            if (!std::isfinite(residual) || std::sqrt(residual / norm) > 1e-8) {
                throw std::runtime_error("Pardiso residual check failed");
            }
        }
        phase(-1);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
