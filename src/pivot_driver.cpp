/**
 * @file pivot_driver.cpp
 * @brief Runs the rank-aware LDLT/LU modes without changing the SPD benchmark.
 *
 * Manufactured loads exercise general multiplication and numerical permutation
 * bookkeeping. Rank-deficient solutions are verified by residual rather than
 * compared to a nonunique manufactured vector. General input values are never
 * passed through the symmetric importer.
 */
#include "pivot_driver.h"
#include "pivot_factor.h"
#include "ordering.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mf {

/**
 * Execute explicit pivoted modes and report their declared numerical policy.
 *
 * PSD is an assertion checked against the computed symmetric inertia estimate.
 * General includes every nonsymmetric sign pattern; positivity of its symmetric
 * part is not used to choose Cholesky. The input and matrix graph are distinct.
 */
int run_pivot_driver(int argc, char** argv) {
    // Parse robust-mode settings independently of SPD-only kernel experiments
    std::string matrix_type = "symmetric";
    std::string ordering = "nd";
    int repeat = 1;
    int threads = 1;
    double rank_tolerance = 1e-12;
    double pivot_tolerance = 0.1;
    int ldlt_block_size = 32;
    int ldlt_supernode_size = 64; // Tuned default; 0 follows --ldlt-block-size (legacy).
    bool fast_pivoting = true, tiled_schur = true, relaxed_supernodes = true;
    bool cached_pivot_rows = true;
    bool pivot_major_cache_refresh = false;
    bool simd_cache_apply = false;
    bool segmented_extend_add = false;
    bool signed_batch_panels = false;
    int signed_batch_width = 16;
    bool indexed_schur = true;
    bool fast_contribution_copy = true;
    bool fast_front_prepare = true;
    bool analyze_only = false;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--matrix-type" && i + 1 < argc) { matrix_type = argv[++i]; }
        else if (option == "--ordering" && i + 1 < argc) { ordering = argv[++i]; }
        else if (option == "--repeat" && i + 1 < argc) { repeat = std::stoi(argv[++i]); }
        else if (option == "--threads" && i + 1 < argc) { threads = std::stoi(argv[++i]); }
        else if (option == "--rank-tolerance" && i + 1 < argc) { rank_tolerance = std::stod(argv[++i]); }
        else if (option == "--pivot-tolerance" && i + 1 < argc) { pivot_tolerance = std::stod(argv[++i]); }
        else if (option == "--ldlt-block-size" && i + 1 < argc) { ldlt_block_size = std::stoi(argv[++i]); }
        else if (option == "--ldlt-supernode-size" && i + 1 < argc) { ldlt_supernode_size = std::stoi(argv[++i]); }
        else if (option == "--ldlt-legacy-pivot-order") { fast_pivoting = false; }
        else if (option == "--ldlt-no-tiled-schur") { tiled_schur = false; }
        else if (option == "--ldlt-no-indexed-schur") { indexed_schur = false; }
        else if (option == "--ldlt-no-fast-contribution-copy") { fast_contribution_copy = false; }
        else if (option == "--ldlt-no-fast-front-prepare") { fast_front_prepare = false; }
        else if (option == "--ldlt-no-pivot-cache") { cached_pivot_rows = false; }
        else if (option == "--ldlt-pivot-major-cache-refresh") { pivot_major_cache_refresh = true; }
        else if (option == "--ldlt-simd-cache-apply") { simd_cache_apply = true; }
        else if (option == "--ldlt-segmented-extend-add") { segmented_extend_add = true; }
        else if (option == "--ldlt-signed-batch-panels") { signed_batch_panels = true; }
        else if (option == "--ldlt-signed-batch-width" && i + 1 < argc) { signed_batch_width = std::stoi(argv[++i]); }
        else if (option == "--ldlt-no-relaxed-supernodes") { relaxed_supernodes = false; }
        else if (option == "--analyze-only") { analyze_only = true; }
        else { throw std::runtime_error("Unsupported or incomplete pivoted-mode option: " + option); }
    }
    if (signed_batch_width < 4 || signed_batch_width > 256) {
        throw std::runtime_error("--ldlt-signed-batch-width must be in [4,256]");
    }
    const bool symmetric = matrix_type == "symmetric" || matrix_type == "psd";
    if ((!symmetric && matrix_type != "general") || repeat < 1 || threads < 1) {
        throw std::runtime_error("Expected matrix-type spd/symmetric/psd/general and positive repeat/threads");
    }
    if (ldlt_block_size < 1 || ldlt_block_size > 256) {
        throw std::runtime_error("--ldlt-block-size must be in [1,256]");
    }
    if (ldlt_supernode_size < 0 || ldlt_supernode_size > 256) {
        throw std::runtime_error("--ldlt-supernode-size must be 0 or in [1,256]");
    }
    if (ordering != "natural" && ordering != "nd") {
        throw std::runtime_error("Invalid ordering; expected nd or natural");
    }

    // Select storage before import: a GENERAL numerical operator is never folded
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto matrix = symmetric ? GeneralMatrix::from_symmetric(Matrix::read_market(argv[1])) :
                                    GeneralMatrix::read_market(argv[1]);
    const auto loaded = Clock::now();
    auto graph = matrix.ordering_graph();
    std::vector<int> order(matrix.n);
    if (ordering == "nd") { order = nested_dissection(graph, threads); }
    else { std::iota(order.begin(), order.end(), 0); }
    const auto ordered = Clock::now();
    const auto permuted = matrix.permute(order, order);
    const auto symbolic = Symbolic::analyze(permuted.ordering_graph(), threads);
    const auto analyzed = Clock::now();
    auto elapsed = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double>(end - begin).count();
    };
    std::cout << std::unitbuf << std::setprecision(9)
              << "solver=pivoted_multifrontal matrix_type=" << matrix_type << " threads=" << threads << '\n'
              << "matrix_n=" << matrix.n << " full_nnz=" << matrix.row.size() << " ordering=" << ordering << '\n'
              << "load_s=" << elapsed(start, loaded) << " graph_ordering_s=" << elapsed(loaded, ordered)
              << " permutation_symbolic_s=" << elapsed(ordered, analyzed) << '\n'
              << "scaling=sqrt_row_column_max rank_tolerance=" << rank_tolerance
              << " pivot_tolerance=" << pivot_tolerance << " rank_method=threshold_front_pivots\n";
    if (analyze_only) { return 0; }

    // Build the same nonconstant manufactured load in original matrix coordinates
    std::vector<double> exact(matrix.n);
    for (int i = 0; i < matrix.n; ++i) { exact[i] = std::sin(0.17 * i) + 0.5 * std::cos(0.031 * i); }
    const auto rhs = matrix.multiply(exact);
    std::vector<double> permuted_rhs(matrix.n);
    for (int i = 0; i < matrix.n; ++i) { permuted_rhs[i] = rhs[order[i]]; }
    PivotFactor factor;
    factor.kind = symmetric ? PivotFactor::Kind::Symmetric : PivotFactor::Kind::General;
    factor.rank_tolerance = rank_tolerance;
    factor.pivot_tolerance = pivot_tolerance;
    factor.threads = threads;
    factor.block_size = ldlt_block_size;
    factor.supernode_size = ldlt_supernode_size;
    factor.fast_pivoting = fast_pivoting;
    factor.tiled_schur = tiled_schur;
    factor.indexed_schur = indexed_schur;
    factor.fast_contribution_copy = fast_contribution_copy;
    factor.fast_front_prepare = fast_front_prepare;
    factor.cached_pivot_rows = cached_pivot_rows;
    factor.pivot_major_cache_refresh = pivot_major_cache_refresh;
    factor.simd_cache_apply = simd_cache_apply;
    factor.segmented_extend_add = segmented_extend_add;
    factor.signed_batch_panels = signed_batch_panels;
    factor.signed_batch_width = signed_batch_width;
    factor.relaxed_supernodes = relaxed_supernodes;
    std::vector<double> totals;
    for (int trial = 0; trial < repeat; ++trial) {
        const auto begin = Clock::now();
        factor.factorize(permuted, symbolic);
        const auto factored = Clock::now();
        if (matrix_type == "psd" && factor.negative_directions != 0) {
            throw std::runtime_error("The declared PSD matrix has negative pivot directions");
        }
        const auto solution = factor.solve(permuted_rhs);
        const auto solved = Clock::now();
        totals.push_back(elapsed(begin, solved));

        // Verify the complete original operator; singular solutions need not equal exact
        std::vector<double> x(matrix.n);
        for (int i = 0; i < matrix.n; ++i) { x[order[i]] = solution[i]; }
        const auto product = matrix.multiply(x);
        long double residual = 0.0L;
        long double norm = 0.0L;
        for (int i = 0; i < matrix.n; ++i) {
            const long double difference = static_cast<long double>(product[i]) - rhs[i];
            residual += difference * difference;
            norm += static_cast<long double>(rhs[i]) * rhs[i];
        }
        const long double relative = norm > 0.0L ? std::sqrt(residual / norm) : std::sqrt(residual);
        std::cout << "trial=" << trial + 1 << " factor_s=" << elapsed(begin, factored)
                  << " solve_s=" << elapsed(factored, solved) << " numeric_total_s=" << totals.back()
                  << " rank_estimate=" << factor.rank << " nullity_estimate=" << matrix.n - factor.rank
                  << " solution_unique=" << int(factor.rank == matrix.n) << " max_front=" << factor.maximum_front
                  << " delayed_candidates=" << factor.delayed_candidates << " pivots_2x2=" << factor.two_by_two_pivots
                  << " relative_residual=" << relative;
        if (symmetric) {
            std::cout << " inertia_positive=" << factor.positive_directions << " inertia_negative=" << factor.negative_directions
                      << " inertia_zero=" << matrix.n - factor.rank;
        }
        std::cout << '\n';
        if (!std::isfinite(relative) || relative > std::max(1e-8, 100.0 * rank_tolerance)) {
            throw std::runtime_error("Pivoted benchmark residual failed");
        }
    }
    std::sort(totals.begin(), totals.end());
    const std::size_t middle = totals.size() / 2;
    const double median = totals.size() % 2 ? totals[middle] : 0.5 * (totals[middle - 1] + totals[middle]);
    if (symmetric) {
        std::cout << "ldlt_block_size=" << ldlt_block_size
                  << " ldlt_supernode_size="
                  << (ldlt_supernode_size == 0 ? ldlt_block_size : ldlt_supernode_size)
                  << " pivot_major_cache_refresh=" << int(pivot_major_cache_refresh)
                  << " simd_cache_apply=" << int(simd_cache_apply)
                  << " signed_batch_panels=" << int(signed_batch_panels)
                  << " signed_batch_width=" << signed_batch_width
                  << " signed_batch_attempts=" << factor.signed_batch_attempts
                  << " signed_batch_pivots=" << factor.signed_batch_pivots
                  << " segmented_extend_add=" << int(segmented_extend_add)
                  << " blocked_panels=" << factor.blocked_panels
                  << " supernodes=" << factor.grouped_fronts
                  << " relaxed_groups=" << factor.relaxed_groups
                  << " fast_pivots=" << factor.fast_path_pivots
                  << " fallback_searches=" << factor.fallback_searches << '\n';
    }
    std::cout << "median_numeric_s=" << median << '\n';
    return 0;
}

} // namespace mf
