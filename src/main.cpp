/**
 * @file main.cpp
 * @brief Standalone multifrontal benchmark with phase timings and residual checks.
 *
 * The driver loads an SPD matrix, orders and analyzes it once, then measures
 * repeated numerical factorizations and solves. Manufactured right-hand sides
 * expose both backward residual and forward error without an external solver.
 */
#include "factor.h"
#include "ordering.h"
#include "pivot_driver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Execute one-thread scalar multifrontal benchmark trials.
 *
 * Import, ordering, permutation and symbolic analysis are timed separately.
 * Each trial refactorizes the same matrix from scratch; triangular solves do
 * not reuse earlier solutions. Residual checks are outside measured solve time.
 * --analyze-only provides a cheap fill/memory-work estimate before factorizing.
 */
int main(int argc, char** argv) {
    try {
        // Parse the small standalone command line before opening the matrix
        if (argc < 2) {
            std::cerr << "Usage: mfsolve matrix.mtx [--repeat N] [--ordering nd|natural]"
                         " [--update-block N|auto]"
                         " [--schur-kernel baseline|packed|compare]"
                         " [--threads N] [--thread-sweep]"
                         " [--huge-pages]"
                         " [--profile-factor off|on|alternate] [--reference-seconds T] [--analyze-only]\n";
            return 2;
        }
        // Select storage and numerical contract before importing any matrix values
        std::string matrix_type = "spd";
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--matrix-type") { matrix_type = argv[i + 1]; }
        }
        if (matrix_type != "spd") { return mf::run_pivot_driver(argc, argv); }
        int         repeat       = 1;
        std::string ordering     = "nd";
        bool        analyze_only = false;
        int         update_block = 0;
        double      reference_s  = 0.0;
        std::string profile_mode = "off";
        std::string schur_mode   = "packed";
        int         thread_count = 1;
        bool        thread_sweep = false;
        bool        huge_pages   = false;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--repeat" && i + 1 < argc) {
                repeat = std::stoi(argv[++i]);
            } else if (option == "--matrix-type" && i + 1 < argc) {
                ++i;
            } else if (option == "--ordering" && i + 1 < argc) {
                ordering = argv[++i];
            } else if (option == "--analyze-only") {
                analyze_only = true;
            } else if (option == "--update-block" && i + 1 < argc) {
                const std::string block = argv[++i];
                update_block = block == "auto" ? 0 : std::stoi(block);
            } else if (option == "--reference-seconds" && i + 1 < argc) {
                reference_s = std::stod(argv[++i]);
            } else if (option == "--profile-factor" && i + 1 < argc) {
                profile_mode = argv[++i];
            } else if (option == "--schur-kernel" && i + 1 < argc) {
                schur_mode = argv[++i];
            } else if (option == "--threads" && i + 1 < argc) {
                thread_count = std::stoi(argv[++i]);
            } else if (option == "--thread-sweep") {
                thread_sweep = true;
            } else if (option == "--huge-pages") {
                huge_pages = true;
            } else {
                throw std::runtime_error("Unknown/incomplete option: " + option);
            }
        }
        if (repeat < 1 || update_block < 0 || update_block > 256 ||
            (ordering != "nd" && ordering != "natural")) {
            throw std::runtime_error("Invalid repeat, update block, or ordering method");
        }
        if (!std::isfinite(reference_s) || reference_s < 0.0) {
            throw std::runtime_error("Reference seconds must be finite and nonnegative");
        }
        if (profile_mode != "off" && profile_mode != "on" && profile_mode != "alternate") {
            throw std::runtime_error("--profile-factor expects off, on, or alternate");
        }
        if (schur_mode != "baseline" && schur_mode != "packed" && schur_mode != "compare") {
            throw std::runtime_error("--schur-kernel expects baseline, packed, or compare");
        }
        if (schur_mode == "compare" && profile_mode == "alternate") {
            throw std::runtime_error("Kernel comparison requires profiling consistently off or on");
        }
        if (thread_count < 1) { throw std::runtime_error("--threads must be positive"); }
        if ((thread_sweep || thread_count > 1) && profile_mode != "off") {
            throw std::runtime_error("Phase profiling is serial; use --profile-factor off for multiple threads");
        }
        if (thread_sweep && schur_mode == "compare") {
            throw std::runtime_error("Thread sweeps require a fixed Schur kernel");
        }
#ifndef _OPENMP
        if (thread_sweep || thread_count > 1) {
            throw std::runtime_error("Build with WITH_OPENMP=1 for multiple threads");
        }
#endif
        using Clock = std::chrono::steady_clock;
        auto seconds = [](Clock::time_point begin, Clock::time_point end) {
            return std::chrono::duration<double>(end - begin).count();
        };
        std::cout << std::unitbuf << std::setprecision(9);
        std::cout << "solver=scalar_multifrontal_cholesky threads=" << thread_count
                  << " thread_sweep=" << int(thread_sweep) << " supernodes=0\n";
        std::cout << "factor_profile_mode=" << profile_mode << '\n';
        std::cout << "schur_kernel_mode=" << schur_mode << '\n';
#if defined(__AVX2__) && defined(__FMA__)
        std::cout << "dense_backend=avx2_fma baseline_tile=4x8 packed_tile=6x8 transpose_dot=avx2_gather\n";
#else
        std::cout << "dense_kernel=portable transpose_dot=portable\n";
#endif

        // Import and prepare the immutable permuted matrix and symbolic plan
        const auto start  = Clock::now();
        const auto matrix = mf::Matrix::read_market(argv[1]);
        const auto loaded = Clock::now();
        std::vector<int> order(matrix.n);
        if (ordering == "nd") {
            order = mf::nested_dissection(matrix, thread_count);
        } else {
            std::iota(order.begin(), order.end(), 0);
        }
        const auto ordered  = Clock::now();
        const auto permuted = matrix.permute(order);
        const auto prepared = Clock::now();
        const auto symbolic = mf::Symbolic::analyze(permuted, thread_count);
        const auto analyzed = Clock::now();
        std::cout << "matrix_n=" << matrix.n << " lower_nnz=" << matrix.row.size() << '\n';
        std::cout << "ordering=" << ordering << " factor_nnz=" << symbolic.row.size()
                  << " max_front=" << symbolic.max_front << " factor_flops=" << symbolic.factor_flops << '\n';
        std::cout << "load_s=" << seconds(start, loaded) << " ordering_s=" << seconds(loaded, ordered)
                  << " permutation_s=" << seconds(ordered, prepared) << " symbolic_s=" << seconds(prepared, analyzed) << '\n';
        std::cout << "factor_storage_mib=" << (symbolic.row.size() * (sizeof(int) + sizeof(double)) / 1048576.0) << '\n';
        if (analyze_only) {
            return 0;
        }

        // Manufacture a nonconstant solution in original coordinates, then permute its RHS
        std::vector<double> exact(matrix.n);
        for (int i = 0; i < matrix.n; ++i) {
            exact[i] = std::sin(0.17 * i) + 0.5 * std::cos(0.031 * i);
        }
        const auto rhs = matrix.multiply(exact);
        std::vector<double> permuted_rhs(matrix.n);
        for (int i = 0; i < matrix.n; ++i) { permuted_rhs[i] = rhs[order[i]]; }
        mf::Factor factor;
        factor.update_block_size = update_block;
        factor.prefer_huge_pages = huge_pages;
        std::vector<double> totals;
        std::vector<double> normal_factor_times;
        std::vector<double> profiled_factor_times;
        std::vector<double> kernel_factor_times[2];
        std::vector<double> kernel_numeric_times[2];
        std::vector<int> thread_counts{thread_count};
        if (thread_sweep) {
            thread_counts.clear();
#ifdef _OPENMP
            for (int count : {1, 2, 4, 8, 16}) {
                if (count <= omp_get_num_procs()) { thread_counts.push_back(count); }
            }
#endif
        }
        std::vector<std::vector<double>> thread_factor_times(thread_counts.size());
        std::vector<std::vector<double>> thread_numeric_times(thread_counts.size());

        // Time complete fresh factorizations and both sparse triangular passes
        for (int trial = 0; trial < repeat * static_cast<int>(thread_counts.size()); ++trial) {
            const int thread_index = trial / repeat;
            factor.threads = thread_counts[thread_index];
            // Alternate in one process to compare identical matrices, orderings and factor capacity
            factor.profile_enabled = profile_mode == "on" || (profile_mode == "alternate" && trial % 2 == 1);
            const bool packed_kernel = schur_mode == "packed" || (schur_mode == "compare" && trial % 2 == 1);
            factor.schur_kernel = packed_kernel ? mf::Factor::SchurKernel::Packed : mf::Factor::SchurKernel::Baseline;
            const auto factor_start = Clock::now();
            factor.factorize(permuted, symbolic);
            const auto factor_end = Clock::now();
            const auto solution   = factor.solve(symbolic, permuted_rhs);
            const auto solve_end  = Clock::now();
            totals.push_back(seconds(factor_start, solve_end));
            auto& factor_times = factor.profile_enabled ? profiled_factor_times : normal_factor_times;
            factor_times.push_back(seconds(factor_start, factor_end));
            kernel_factor_times[packed_kernel].push_back(seconds(factor_start, factor_end));
            kernel_numeric_times[packed_kernel].push_back(totals.back());
            thread_factor_times[thread_index].push_back(seconds(factor_start, factor_end));
            thread_numeric_times[thread_index].push_back(totals.back());

            // Verify against the original operator, outside the timed solve region
            std::vector<double> original_x(matrix.n);
            for (int i = 0; i < matrix.n; ++i) { original_x[order[i]] = solution[i]; }
            const auto product = matrix.multiply(original_x);
            long double residual_squared = 0.0L;
            long double rhs_squared      = 0.0L;
            long double error_squared    = 0.0L;
            long double exact_squared    = 0.0L;
            for (int i = 0; i < matrix.n; ++i) {
                const long double r = static_cast<long double>(product[i]) - rhs[i];
                const long double e = static_cast<long double>(original_x[i]) - exact[i];
                residual_squared += r * r;
                rhs_squared      += static_cast<long double>(rhs[i]) * rhs[i];
                error_squared    += e * e;
                exact_squared    += static_cast<long double>(exact[i]) * exact[i];
            }
            const double residual = std::sqrt(residual_squared / rhs_squared);
            const double error    = std::sqrt(error_squared / exact_squared);
            std::cout << "trial=" << trial + 1 << " update_block=" << factor.update_block_size_used
                      << " update_block_mode=" << (update_block == 0 ? "auto" : "explicit") << " factor_s=" << seconds(factor_start, factor_end)
                      << " solve_s=" << seconds(factor_end, solve_end) << " numeric_total_s=" << totals.back()
                      << " relative_residual=" << residual << " relative_solution_error=" << error
                      << " peak_contribution_mib=" << factor.peak_contribution_bytes / 1048576.0
                      << " inherited_fronts=" << factor.inherited_fronts
                      << " minimum_pivot=" << factor.minimum_pivot
                      << " schur_kernel=" << (packed_kernel ? "packed" : "baseline")
                      << " threads=" << factor.threads
                      << " threads_used=" << factor.threads_used
                      << " value_alignment=" << (reinterpret_cast<std::uintptr_t>(factor.value.data()) % 64)
                      << " huge_page_hints_accepted=" << factor.huge_page_hints_accepted
                      << " process_anon_huge_kib=" << mf::memory::anonymous_huge_pages_kib()
                      << " factor_profile=" << int(factor.profile_enabled);
            if (factor.profile_enabled) {
                std::cout << " profile_memory_s=" << factor.phase_seconds[mf::Factor::Memory]
                          << " profile_assembly_s=" << factor.phase_seconds[mf::Factor::Assembly]
                          << " profile_pivot_correction_s=" << factor.phase_seconds[mf::Factor::PivotCorrection]
                          << " profile_elimination_s=" << factor.phase_seconds[mf::Factor::Elimination]
                          << " profile_schur_s=" << factor.phase_seconds[mf::Factor::SchurUpdate]
                          << " profile_other_s=" << factor.phase_seconds[mf::Factor::Other];
            }
            std::cout << '\n';
            if (!std::isfinite(residual) || residual > 1e-8 || !std::isfinite(error)) {
                throw std::runtime_error("Solution verification failed");
            }
        }

        // Report every trial above and a conventional median without dropping warm-up runs
        auto median_time = [](std::vector<double>& times) {
            std::sort(times.begin(), times.end());
            const std::size_t middle = times.size() / 2;
            return times.size() % 2 ? times[middle] : 0.5 * (times[middle - 1] + times[middle]);
        };
        const double median = median_time(totals);
        const double preparation = seconds(loaded, analyzed);
        std::cout << "median_numeric_s=" << median << " min_numeric_s=" << totals.front()
                  << " median_preparation_plus_numeric_s=" << preparation + median;
        if (reference_s > 0.0) {
            std::cout << " reference_femaster_s=" << reference_s
                      << " numeric_vs_reference=" << median / reference_s;
        }
        std::cout << '\n';
        if (!normal_factor_times.empty() && !profiled_factor_times.empty()) {
            const double normal   = median_time(normal_factor_times);
            const double profiled = median_time(profiled_factor_times);
            std::cout << "unprofiled_median_factor_s=" << normal << " profiled_median_factor_s=" << profiled
                      << " profiling_time_ratio=" << profiled / normal << '\n';
        }
        if (!kernel_factor_times[0].empty() && !kernel_factor_times[1].empty()) {
            const double baseline_factor = median_time(kernel_factor_times[0]);
            const double packed_factor   = median_time(kernel_factor_times[1]);
            std::cout << "baseline_median_factor_s=" << baseline_factor << " packed_median_factor_s=" << packed_factor
                      << " kernel_factor_speedup=" << baseline_factor / packed_factor
                      << " baseline_median_numeric_s=" << median_time(kernel_numeric_times[0])
                      << " packed_median_numeric_s=" << median_time(kernel_numeric_times[1]) << '\n';
        }
        const double first_factor = median_time(thread_factor_times[0]);
        for (std::size_t index = 0; index < thread_counts.size(); ++index) {
            const double time = median_time(thread_factor_times[index]);
            std::cout << "thread_summary=" << thread_counts[index] << " median_factor_s=" << time
                      << " median_numeric_s=" << median_time(thread_numeric_times[index])
                      << " factor_speedup_vs_first=" << first_factor / time << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
