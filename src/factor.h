/**
 * @file factor.h
 * @brief Numerical multifrontal Cholesky panels and scalar CSC substitution.
 *
 * Factorization merges exact scalar-tree chains into multi-pivot fronts,
 * consumes their child Schur complements and exports one contribution each.
 * Persistent L columns are separate from temporary contribution storage.
 */
#pragma once

#include "symbolic.h"
#include "memory.h"

#include <array>

namespace mf {

/**
 * @brief Owns scalar CSC factors computed through exact amalgamated fronts.
 *
 * All arithmetic is double precision. Positive finite pivots are required;
 * failure is reported rather than hidden through shifts or pivot replacement.
 * The caller retains the matching Symbolic object for factorization and solves.
 * Contributions are transient: a matching parent inherits the backing buffer,
 * otherwise it assembles and releases the child. Every panel is fully summed
 * before leading-block POTRF and rectangular TRSM. Its complete Schur update
 * is exported before a branch or index-set change.
 * Group width follows update_block_size. Repeated factorization reuses factor
 * capacity, not numerical results. Scalar CSC storage and solve APIs are kept.
 */
struct Factor {
    // PivotCorrection measures panel TRSM; Elimination measures leading POTRF.
    // Nonoverlapping numerical timing categories; disabled specializations of the inner
    // algorithm compile out all clock reads rather than timing empty scopes.
    enum ProfilePhase { Memory, Assembly, PivotCorrection, Elimination, SchurUpdate, Other, PhaseCount };

    // The packed mode retains the baseline for small fronts and low update ranks
    enum class SchurKernel { Baseline, Packed };

    // Numerical columns are initialized by their owning worker before use
    memory::UninitializedValues value;

    // Numerical diagnostics from the last factorization. inherited_fronts counts
    // matching scalar links, including pivots absorbed into assembled panels.
    std::size_t         peak_contribution_bytes = 0;
    double              minimum_pivot          = 0.0;
    std::size_t         inherited_fronts        = 0;
    int                 update_block_size_used = 0;

    // Maximum exact pivot group and Schur update rank, independent of thread
    // count. The common default is 128; explicit widths lie in [1,256].
    int         update_block_size = 128;
    SchurKernel schur_kernel      = SchurKernel::Packed;

    // One OpenMP team serves subtree tasks and large Schur tiles. Profiling
    // remains serial because overlapping thread intervals are not wall phases.
    // The same count controls independent-subtree SPD substitution.
    int threads = 1;
    int threads_used = 1;
    bool        prefer_huge_pages        = false;
    std::size_t huge_page_hints_accepted = 0;

    // Optional per-factorization wall-clock totals. Other includes traversal,
    // timer bookkeeping and final temporary-container destruction.
    bool                           profile_enabled = false;
    std::array<double, PhaseCount> phase_seconds{};

    // Numerical groups are inferred only from scalar symbolic suffix matches.
    // These counters distinguish actual assembled panels from scalar columns.
    std::size_t assembled_fronts = 0;
    std::size_t merged_pivots    = 0;

    // Assemble exact pivot panels in tree order and preserve scalar CSC factors
    void factorize(const Matrix& matrix, const Symbolic& symbolic);

    // Solve in permuted coordinates using the stored lower factor and transpose.
    // Exact consecutive chains share forward row traversal in groups of eight.
    // Large single-root plans merge task-private RHS contributions before the
    // separator solve, avoiding overlapping forward scatters into shared state.
    std::vector<double> solve(const Symbolic& symbolic, const std::vector<double>& rhs) const;

private:
    // Instantiate the panel algorithm with or without phase clock reads
    template<bool Profile, bool Parallel>
    void factorize_impl(const Matrix& matrix, const Symbolic& symbolic);
};

} // namespace mf
