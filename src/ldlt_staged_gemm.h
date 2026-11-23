/**
 * @file ldlt_staged_gemm.h
 * @brief Experimental GEMM-style Schur updates with temporary dense output tiles.
 *
 * No changes to the row-packed frontal matrix, child contribution storage,
 * scalar/2x2 threshold pivoting, or physical inheritance offsets.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace mf::dense {

struct StagedGemmProfile {
    double pack_s = 0.0;       // Packing L/R once per complete Schur flush.
    double load_s = 0.0;       // Gathering C into scratch output tiles.
    double gemm_s = 0.0;       // Dense GEMM arithmetic (internal or CBLAS).
    double store_s = 0.0;      // Scattering the updated lower-triangle C.
    std::size_t tiles = 0;
};

// C(active,active) -= L(active,:) * R(active,:)^T, with the signed D
// contribution already absorbed in R. 'active' is sorted and may have holes.
// The physical frontal buffer remains an ordinary row-packed lower triangle;
// origin offsets local IDs into an inherited physical storage allocation.
// Diagonal output tiles update only their meaningful lower triangle.
void rank_k_ldlt_staged_gemm(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed,
    StagedGemmProfile* timings = nullptr);

} // namespace mf::dense
