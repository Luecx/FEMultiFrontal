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
    int size = 1;
    std::array<int, 2> rows{-1, -1};
    std::array<int, 2> columns{-1, -1};
    std::array<double, 4> diagonal{};
    std::vector<int> lower_rows;
    std::vector<std::array<double, 2>> lower_values;
    std::vector<int> upper_columns;
    std::vector<std::array<double, 2>> upper_values;
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
struct PivotFactor {
    enum class Kind { Symmetric, General };

    // Numerical policy in equilibrated coordinates
    Kind   kind            = Kind::Symmetric;
    double rank_tolerance  = 1e-12;
    double pivot_tolerance = 0.1;
    int    threads         = 1;

    // Accepted factors and threshold-based diagnostics
    std::vector<PivotBlock> blocks;
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

    // Recover one basic solution and reject an incompatible original-system residual
    std::vector<double> solve(const std::vector<double>& rhs) const;
};

} // namespace mf
