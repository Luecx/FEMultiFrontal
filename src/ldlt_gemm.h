/** @file ldlt_gemm.h
 * Dense scratch-tile GEMM-like Schur update of packed indefinite fronts.
 */
#pragma once
#include <vector>
namespace mf::dense {
// Updates only the live lower triangle, leaving delayed/deflated holes
// and entries outside the inherited origin entirely unchanged.
void rank_k_ldlt_gemm(int origin, int dim, const std::vector<int>& active,
                      int rank, const double* left, const double* right,
                      double* packed);
}
