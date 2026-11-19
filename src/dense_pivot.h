/**
 * @file dense_pivot.h
 * @brief Dense symmetric rank-one/rank-two updates for dynamic LDLT fronts.
 *
 * PivotFactor owns pivot selection, scaling and active index sets. This module
 * subtracts one accepted pivot's correction from the active lower triangle and
 * restores its mirror in the full row-major front. The panel kernel applies
 * a sequence of accepted scalar corrections to an inherited trailing front.
 * PivotFactor materializes candidate entries before threshold tests.
 */
#pragma once

#include <array>

namespace mf::dense {

// Subtract lower[row] * upper[column] from sorted active lower-triangle entries.
// Rows and columns are sorted, distinct local indices. The physical row stride
// may exceed the live width of an inherited front. The lower array has row_count
// entries and upper arrays use local column indexing. Rank lies in [1,2].
// Mirror updated off-diagonal entries and reject nonfinite results. An existing
// OpenMP team can share the update for sufficiently large active fronts.
void symmetric_pivot_update(
    int                          stride,
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

// Complete a panel of accepted scalar LDLT corrections in acceptance order.
// Each coefficient column spans the physical stride. The live suffix begins
// at origin in both dimensions of the full row-major frontal buffer.
void flush_scalar_pivot_updates(
    int stride, int origin, int end, int count, const double* lower, const double* upper, double* values
);

} // namespace mf::dense
