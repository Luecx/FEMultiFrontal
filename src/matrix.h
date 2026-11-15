/**
 * @file matrix.h
 * @brief Owns symmetric sparse input independently of FEMaster and external libraries.
 *
 * Only the lower triangle is stored in compressed columns. Matrix Market input
 * validation and symmetric multiplication belong here; fill and factors belong
 * to the symbolic and numerical solver modules.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mf {

/**
 * @brief Symmetric real matrix stored as a lower-triangular CSC array.
 *
 * Column pointers use size_t for large nonzero counts; row indices are signed
 * 32-bit integers. Rows within each column are sorted and duplicate-free.
 * Missing entries represent zero. Values own their storage and retain original
 * physical scaling; no diagonal shifting or equilibration is implicit.
 */
struct Matrix {
    int                      n = 0;
    std::vector<std::size_t> ptr;
    std::vector<int>         row;
    std::vector<double>      value;

    // Import symmetric coordinate input, validating both halves of GENERAL files
    static Matrix read_market(const std::string& path);

    // Apply both symmetric halves without materializing a full sparse matrix
    std::vector<double> multiply(const std::vector<double>& x) const;

    // Form P A P^T; order[new_index] gives the original index
    Matrix permute(const std::vector<int>& order) const;
};

} // namespace mf
