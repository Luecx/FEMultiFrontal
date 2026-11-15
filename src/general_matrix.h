/**
 * @file general_matrix.h
 * @brief Stores the complete numerical operator for general sparse systems.
 *
 * Unlike Matrix's symmetric lower triangle, this CSC representation preserves
 * both halves independently. It supplies general multiplication and independent
 * row/column permutations for LU, plus a structural graph for ordering only.
 */
#pragma once

#include "matrix.h"
#include "memory.h"

namespace mf {

/**
 * @brief Owns a real square sparse matrix without a symmetry assumption.
 *
 * Columns are sorted and duplicate-free; exact zeros are omitted. GENERAL
 * Matrix Market values are never averaged or mirrored. SYMMETRIC input is
 * explicitly expanded, and conversion from Matrix preserves its canonical
 * symmetric values. Numerical values use cache-line-aligned owned storage.
 */
struct GeneralMatrix {
    int                      n = 0;
    std::vector<std::size_t> ptr;
    std::vector<int>         row;
    memory::Values          value;

    // Full numerical import or explicit expansion of a canonical symmetric matrix
    static GeneralMatrix read_market(const std::string& path);
    static GeneralMatrix from_symmetric(const Matrix& matrix);

    // Apply A, without mirroring off-diagonal entries
    std::vector<double> multiply(const std::vector<double>& x) const;

    // Row and column orders both use new-to-original indexing
    GeneralMatrix permute(const std::vector<int>& rows, const std::vector<int>& columns) const;

    // Undirected union of structural A and A^T. Unit values are graph metadata,
    // not a symmetrization of the numerical operator used by LU.
    Matrix ordering_graph() const;
};

} // namespace mf
