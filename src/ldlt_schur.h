/**
 * @file ldlt_schur.h
 * @brief Cache-tiled asymmetric rank-k updates of packed symmetric LDLT fronts.
 */
#pragma once
#include <vector>

namespace mf::dense {
// C(active,active) -= left(active,:) * right(active,:)^T, keeping inactive
// packed entries untouched. Both panels are pivot-major with pitch dim.
// The physical packed triangle begins at origin; active is sorted and local.
void rank_k_ldlt_tiled(int origin, int dim, const std::vector<int>& active,
                       int rank, const double* left, const double* right,
                       double* packed);
}
