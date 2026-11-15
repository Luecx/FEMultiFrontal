/**
 * @file dense.h
 * @brief Raw dense arithmetic used by scalar multifrontal Schur updates.
 *
 * Dense contribution matrices store their lower triangles in row-packed form.
 * Scratch belongs to the caller; kernels introduce no BLAS or FEMaster dependency.
 * Scalar updates can be applied immediately or batched along inherited-buffer
 * chains. The AVX2/FMA rank-k kernel changes neither pivot nor tree ownership.
 */
#pragma once

#include <vector>

namespace mf::dense {

// Compute C -= x x^T in row-packed lower storage, including the diagonal
void rank_one_subtract(int n, const double* x, double* packed, int origin = 0);

// Compute C -= X X^T using k column pointers to n entries each. origin skips
// previously eliminated leading rows/columns of an inherited packed matrix.
void rank_k_subtract(int n, int k, const double* const* columns, double* packed, int origin);

// Alternative register tile and packed input panels for measured kernel comparisons.
// The packed variant borrows caller-owned scratch and includes panel construction.
void rank_k_subtract_wide(int n, int k, const double* const* columns, double* packed, int origin);
void rank_k_subtract_packed(
    int                  n,
    int                  k,
    const double* const* columns,
    double*              packed,
    int                  origin,
    std::vector<double>& scratch
);
void rank_k_subtract_packed6(
    int                  n,
    int                  k,
    const double* const* columns,
    double*              packed,
    int                  origin,
    std::vector<double>& scratch
);

// Dot product against indirectly addressed vector entries for transpose substitution
double indexed_dot(int n, const int* indices, const double* values, const double* x);

} // namespace mf::dense
