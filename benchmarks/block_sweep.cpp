/**
 * @file block_sweep.cpp
 * @brief Compares Schur batch widths on one immutable ordering and symbolic plan.
 *
 * Trials rotate the order of widths to reduce temperature and clock bias.
 * The first sample of each width is a warm-up. Residuals use the original matrix.
 * Optional serial profiling reports Schur wall time separately.
 */
#include "factor.h"
#include "ordering.h"
#include "load_matrix.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr << "Usage: block_sweep matrix.mtx threads repeats [profile|off] [32,64,128,256]\n";
            return 2;
        }
        const int threads = std::stoi(argv[2]);
        const int repeat = std::stoi(argv[3]);
        const bool profile = argc >= 5 && std::string(argv[4]) == "profile";
        if (threads < 1 || repeat < 2 || (profile && threads != 1)) {
            throw std::runtime_error("Require positive threads, repeats >= 2, serial profiling");
        }
        if (argc >= 5 && std::string(argv[4]) != "profile" && std::string(argv[4]) != "off") {
            throw std::runtime_error("Expected profile or off");
        }
        std::vector<int> widths{32, 64, 128, 256};
        if (argc == 6) {
            widths.clear();
            std::string list = argv[5];
            std::replace(list.begin(), list.end(), ',', ' ');
            std::istringstream input(list);
            int width = 0;
            while (input >> width) {
                if (width < 1 || width > 256) { throw std::runtime_error("Invalid batch width"); }
                widths.push_back(width);
            }
            if (!input.eof() || widths.empty()) { throw std::runtime_error("Invalid batch-width list"); }
        }
        using Clock = std::chrono::steady_clock;
        std::cout << std::unitbuf << std::setprecision(9);
        const auto matrix = load_benchmark_matrix(argv[1]);
        const auto start = Clock::now();
        // An optional exported permutation isolates ordering quality from
        // numerical kernels; its generation time is reported by its producer.
        const char* order_path = std::getenv("MF_BENCH_ORDER");
        const auto order = [&]() {
            if (!order_path) { return mf::nested_dissection(matrix, threads); }
            std::vector<int> result(matrix.n);
            std::ifstream file(order_path, std::ios::binary);
            file.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(int));
            if (!file || file.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("Invalid external ordering size");
            }
            std::vector<bool> seen(matrix.n, false);
            for (int value : result) {
                if (value < 0 || value >= matrix.n || seen[value]) {
                    throw std::runtime_error("Invalid external permutation");
                }
                seen[value] = true;
            }
            return result;
        }();
        const auto permuted = matrix.permute(order);
        const auto symbolic = mf::Symbolic::analyze(permuted, threads);
        std::cout << "matrix_n=" << matrix.n << " threads=" << threads
                  << " ordering=" << (order_path ? "external" : "nd")
                  << " preparation_s=" << std::chrono::duration<double>(Clock::now()-start).count()
                  << " factor_nnz=" << symbolic.row.size() << " max_front=" << symbolic.max_front << '\n';
        std::vector<double> exact(matrix.n);
        for (int i = 0; i < matrix.n; ++i) {
            exact[i] = std::sin(0.17 * i) + 0.5 * std::cos(0.031 * i);
        }
        const auto rhs = matrix.multiply(exact);
        std::vector<double> permuted_rhs(matrix.n);
        for (int i = 0; i < matrix.n; ++i) { permuted_rhs[i] = rhs[order[i]]; }
        mf::Factor factor;
        factor.threads = threads;
        factor.profile_enabled = profile;
        for (int trial = 0; trial < repeat; ++trial) {
            for (std::size_t variant = 0; variant < widths.size(); ++variant) {
                factor.update_block_size = widths[(trial + variant) % widths.size()];
                const auto begin = Clock::now();
                factor.factorize(permuted, symbolic);
                const auto factored = Clock::now();
                const auto solution = factor.solve(symbolic, permuted_rhs);
                const auto solved = Clock::now();
                std::vector<double> original(matrix.n);
                for (int i = 0; i < matrix.n; ++i) { original[order[i]] = solution[i]; }
                const auto product = matrix.multiply(original);
                long double residual = 0, norm = 0, error = 0, exact_norm = 0;
                for (int i = 0; i < matrix.n; ++i) {
                    const long double r = static_cast<long double>(product[i]) - rhs[i];
                    const long double e = static_cast<long double>(original[i]) - exact[i];
                    residual += r * r;
                    norm += static_cast<long double>(rhs[i]) * rhs[i];
                    error += e * e;
                    exact_norm += static_cast<long double>(exact[i]) * exact[i];
                }
                const double relative = std::sqrt(residual / norm);
                std::cout << "trial=" << trial + 1 << " warmup=" << (trial == 0)
                          << " threads_used=" << factor.threads_used
                          << " update_block=" << factor.update_block_size
                          << " factor_s=" << std::chrono::duration<double>(factored-begin).count()
                          << " solve_s=" << std::chrono::duration<double>(solved-factored).count()
                          << " relative_residual=" << relative
                          << " relative_solution_error=" << std::sqrt(error / exact_norm);
                if (profile) { std::cout << " schur_s=" << factor.phase_seconds[mf::Factor::SchurUpdate]; }
                std::cout << '\n';
                if (!std::isfinite(relative) || relative > 1e-8) {
                    throw std::runtime_error("Residual check failed");
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
