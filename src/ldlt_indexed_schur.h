/**
 * @file ldlt_indexed_schur.h
 * @brief Packed L/R panels and index-hole-safe AVX2 updates of LDLT fronts.
 */
#pragma once
#include <vector>

namespace mf::dense {

// C(active,active) -= left(active,:) * right(active,:)^T.
// left/right are pivot-major, each of pitch dim, and right has already been
// transformed by the signed 1x1 / indefinite 2x2 pivot D blocks.
// C remains in the original inherited row-packed lower-triangle layout.
// The active index set is sorted and may contain delayed-pivot holes.
// No caller-owned persistent factor, matrix, or pivot state is modified.
void rank_k_ldlt_indexed_packed(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed);

// Experimental 6-row x 8-column microtile, otherwise same semantics as
// rank_k_ldlt_indexed_packed (4-row x 8-column).
void rank_k_ldlt_indexed_packed6(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed);

} // namespace mf::dense
