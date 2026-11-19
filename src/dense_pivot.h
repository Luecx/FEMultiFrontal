/**
 * @file dense_pivot.h
 * @brief Dense symmetric rank-one/rank-two updates for dynamic LDLT fronts.
 *
 * PivotFactor owns pivot selection, scaling and active index sets. This module
 * subtracts one accepted pivot's correction from the active lower triangle and
 * restores its mirror in the full row-major front. No updates are deferred, so
 * subsequent threshold and rank decisions observe the complete current matrix.
 */
#pragma once

#include <array>

namespace mf::dense {

// Subtract lower[row] * upper[column] from sorted active lower-triangle entries.
// Rows/columns are distinct local indices in [0,dimension); lower has row_count
// entries and upper arrays use full local column indexing. Rank lies in [1,2];
// upper1 is ignored for rank one. Mirror every updated off-diagonal entry and
// reject nonfinite results. Pivot data and inactive entries remain unchanged.
void symmetric_pivot_update(
    int                          dimension,
    int                          rank,
    int                          row_count,
    const int*                   rows,
    const std::array<double, 2>*  lower,
    int                          column_count,
    const int*                   columns,
    const double*                upper0,
    const double*                upper1,
    double*                      values
);

} // namespace mf::dense
