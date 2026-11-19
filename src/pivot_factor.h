/**
 * @file pivot_factor.h
 * @brief Defines rank-aware multifrontal LDLT/LU factors with delayed pivots.
 *
 * This robust path is separate from fast SPD Cholesky. It stores numerical
 * pivot blocks and independent row/column identities, allowing delayed variables
 * to change frontal patterns without pretending they match the SPD factor plan.
 */
#pragma once

#include "general_matrix.h"
#include "symbolic.h"

#include <array>
#include <memory>
#include <memory_resource>

namespace mf {

/**
 * @brief One accepted scalar or two-variable numerical elimination block.
 *
 * Row and column IDs refer to the structurally permuted input coordinates.
 * A symmetric block is D in LDLT; a general block belongs to U. Lower values
 * store normalized multipliers. General upper entries retain the raw pivot-row
 * coupling; symmetric substitution instead uses D and the transpose of L.
 */
struct PivotBlock {
    explicit PivotBlock(std::pmr::memory_resource* resource = std::pmr::get_default_resource())
        : lower_rows(resource), lower_values(resource),
          upper_columns(resource), upper_values(resource) {}

    int size = 1;
    std::array<int, 2> rows{-1, -1};
    std::array<int, 2> columns{-1, -1};
    std::array<double, 4> diagonal{};
    std::pmr::vector<int> lower_rows;
    std::pmr::vector<std::array<double, 2>> lower_values;
    std::pmr::vector<int> upper_columns;
    std::pmr::vector<std::array<double, 2>> upper_values;
};

/**
 * @brief Sparse numerical pivot factorization over a symbolic assembly forest.
 *
 * Fully summed candidates may be delayed to a parent. Symmetric mode uses
 * congruent scaling and 1-by-1/2-by-2 threshold pivots. General mode chooses
 * scalar row/column pivots independently. Only root residual blocks below the
 * scaled rank threshold are deflated. Numerical rank is an estimate under that
 * policy, not a singular-value certificate. No diagonal regularization is used.
 *
 * The input matrix must outlive solve calls. Consistent singular systems return
 * a basic solution with free scaled coordinates zero, not a minimum-norm solution.
 * Inconsistent loads fail an independent residual check against the original A.
 */
struct FastSpdState;

struct PivotFactor {
    enum class Kind { Symmetric, General };

    PivotFactor() = default;
    PivotFactor(const PivotFactor&) = default;
    PivotFactor(PivotFactor&&) noexcept = default;

    // Copy-and-swap is required: ordinary memberwise assignment could free
    // the old PMR arena before replacing its still-live PivotBlock vectors.
    PivotFactor& operator=(PivotFactor other) noexcept {
        using std::swap;
        swap(kind, other.kind);
        swap(rank_tolerance, other.rank_tolerance);
        swap(pivot_tolerance, other.pivot_tolerance);
        swap(threads, other.threads);
        swap(fast_spd, other.fast_spd);
        swap(blocks, other.blocks);
        swap(spd_blocks_materialized, other.spd_blocks_materialized);
        swap(rank, other.rank);
        swap(positive_directions, other.positive_directions);
        swap(negative_directions, other.negative_directions);
        swap(two_by_two_pivots, other.two_by_two_pivots);
        swap(delayed_candidates, other.delayed_candidates);
        swap(maximum_front, other.maximum_front);
        swap(minimum_pivot, other.minimum_pivot);
        swap(matrix, other.matrix);
        swap(row_scale, other.row_scale);
        swap(column_scale, other.column_scale);
        swap(ready, other.ready);
        return *this;
    }

    // Numerical policy in equilibrated coordinates
    Kind   kind            = Kind::Symmetric;
    double rank_tolerance  = 1e-12;
    double pivot_tolerance = 0.1;
    int    threads         = 1;

    // Fast-state lifetime must extend beyond the PMR-backed pivot blocks.
    // Kept before blocks so destruction releases blocks before their arena.
    std::shared_ptr<FastSpdState> fast_spd;

    // Accepted factors and threshold-based diagnostics. For an accepted SPD
    // fast path, blocks contain the pivot trace (rows, columns, diagonals),
    // but multipliers are retained only in fast_spd->factor until explicitly
    // requested with materialize_spd_blocks(). General/indefinite factor blocks
    // are fully populated as before.
    std::vector<PivotBlock> blocks;
    bool spd_blocks_materialized = false;
    int rank                 = 0;
    int positive_directions  = 0;
    int negative_directions  = 0;
    int two_by_two_pivots    = 0;
    std::size_t delayed_candidates = 0;
    std::size_t maximum_front      = 0;
    double minimum_pivot          = 0.0;

    // Matrix lifetime and scaling in equation and variable spaces
    const GeneralMatrix* matrix = nullptr;
    std::vector<double> row_scale;
    std::vector<double> column_scale;
    bool ready = false;

    // Numerical assembly follows the supplied structural forest, retaining delayed IDs
    void factorize(const GeneralMatrix& input, const Symbolic& symbolic);

    // Populate per-pivot lower rows and multipliers on demand for consumers
    // that inspect the full LDLT factor. No-op for robust LDLT/LU.
    // SPD solve and pivot diagnostics never require this materialization.
    void materialize_spd_blocks();

    // Recover one basic solution and reject an incompatible original-system residual
    std::vector<double> solve(const std::vector<double>& rhs) const;
};

} // namespace mf
