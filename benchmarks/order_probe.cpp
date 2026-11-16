/**
 * @file order_probe.cpp
 * @brief Measures ordering, exact fill and predicted Cholesky work before factoring.
 *
 * Optional permutation export allows a numerical comparison with an immutable
 * plan. Only graph ordering is changed; matrix values remain untouched.
 */
#include "load_matrix.h"
#include "ordering.h"
#include "symbolic.h"

#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: order_probe matrix.csc threads order.bin\n";
            return 2;
        }
        const int threads = std::stoi(argv[2]);
        if (threads < 1) { throw std::runtime_error("Positive thread count required"); }
        const auto matrix = load_benchmark_matrix(argv[1]);
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        const auto order = mf::nested_dissection(matrix, threads);
        const auto ordered = Clock::now();
        const auto permuted = matrix.permute(order);
        const auto prepared = Clock::now();
        const auto symbolic = mf::Symbolic::analyze(permuted, threads);
        const auto analyzed = Clock::now();
        std::ofstream file(argv[3], std::ios::binary);
        file.write(reinterpret_cast<const char*>(order.data()), order.size() * sizeof(int));
        file.close();
        if (!file) { throw std::runtime_error("Failed to write permutation"); }
        std::cout << std::setprecision(12)
                  << "n=" << matrix.n << " ordering_s="
                  << std::chrono::duration<double>(ordered-start).count()
                  << " permutation_s=" << std::chrono::duration<double>(prepared-ordered).count()
                  << " symbolic_s=" << std::chrono::duration<double>(analyzed-prepared).count()
                  << " factor_nnz=" << symbolic.row.size()
                  << " max_front=" << symbolic.max_front
                  << " factor_flops=" << symbolic.factor_flops << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
