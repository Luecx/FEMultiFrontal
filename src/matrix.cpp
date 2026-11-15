/**
 * @file matrix.cpp
 * @brief Implements validated Matrix Market import and symmetric CSC operations.
 *
 * GENERAL input is checked for numerical symmetry before averaging matching
 * halves. SYMMETRIC input supplies one triangle. Duplicate coordinates are
 * summed before compression. No solver or FEMaster types are required.
 */
#include "matrix.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace mf {

/**
 * Read a real coordinate Matrix Market file into canonical lower CSC storage.
 *
 * GENERAL files must contain matching symmetric halves, up to 1e-12 times
 * the largest input entry. Duplicate coordinates are summed independently in
 * each half. Exact zeros are removed after symmetry validation. Input indices
 * are checked before conversion from one-based to zero-based coordinates.
 */
Matrix Matrix::read_market(const std::string& path) {
    // Read once into memory so parsing does not use formatted stream extraction
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open matrix: " + path);
    }
    const std::string input((std::istreambuf_iterator<char>(file)), {});
    const auto header_end = input.find('\n');
    const auto header     = input.substr(0, header_end);
    const bool symmetric  = header == "%%MatrixMarket matrix coordinate real symmetric" ||
                            header == "%%MatrixMarket matrix coordinate real symmetric\r";
    const bool general    = header == "%%MatrixMarket matrix coordinate real general" ||
                            header == "%%MatrixMarket matrix coordinate real general\r";
    if ((!general && !symmetric) || header_end == std::string::npos) {
        throw std::runtime_error("Expected Matrix Market real coordinate GENERAL or SYMMETRIC");
    }

    // Skip whitespace and Matrix Market comment lines between numeric fields
    const char* cursor = input.data() + header_end + 1;
    const char* end    = input.data() + input.size();
    auto skip = [&]() {
        while (cursor < end) {
            if (*cursor == '%') {
                while (cursor < end && *cursor != '\n') { ++cursor; }
            } else if (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t') {
                ++cursor;
            } else {
                break;
            }
        }
    };
    auto number = [&](auto& value) {
        skip();
        const auto result = std::from_chars(cursor, end, value);
        if (result.ec != std::errc{} || result.ptr == cursor) {
            throw std::runtime_error("Invalid or truncated Matrix Market numeric field");
        }
        cursor = result.ptr;
    };

    // Keep orientation while sorting so GENERAL symmetry can be checked
    int         rows  = 0;
    int         cols  = 0;
    std::size_t count = 0;
    number(rows);
    number(cols);
    number(count);
    if (rows <= 0 || rows != cols || count > input.size() / 5) {
        throw std::runtime_error("Invalid square matrix dimensions or entry count");
    }
    std::vector<std::tuple<int, int, int, double>> entries;
    entries.reserve(count);
    double largest = 0.0;
    for (std::size_t k = 0; k < count; ++k) {
        int    r = 0;
        int    c = 0;
        double v = 0.0;
        number(r);
        number(c);
        number(v);
        if (r < 1 || r > rows || c < 1 || c > cols || !std::isfinite(v)) {
            throw std::runtime_error("Matrix entry has invalid indices or a nonfinite value");
        }
        largest = std::max(largest, std::abs(v));
        entries.emplace_back(std::min(r, c) - 1, std::max(r, c) - 1, r < c ? 1 : 0, v);
    }
    skip();
    if (cursor != end) {
        throw std::runtime_error("Unexpected data after declared Matrix Market entries");
    }
    std::sort(entries.begin(), entries.end());

    // Sum duplicate coordinates and compare corresponding symmetric entries
    Matrix matrix;
    matrix.n = rows;
    matrix.ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    matrix.row.reserve(symmetric ? count : count / 2 + rows);
    matrix.value.reserve(matrix.row.capacity());
    for (std::size_t i = 0; i < entries.size();) {
        const int c = std::get<0>(entries[i]);
        const int r = std::get<1>(entries[i]);
        double lower = 0.0;
        double upper = 0.0;
        do {
            if (std::get<2>(entries[i]) == 0) {
                lower += std::get<3>(entries[i]);
            } else {
                upper += std::get<3>(entries[i]);
            }
            ++i;
        } while (i < entries.size() && std::get<0>(entries[i]) == c && std::get<1>(entries[i]) == r);
        if (!std::isfinite(lower) || !std::isfinite(upper)) {
            throw std::runtime_error("Nonfinite sum of duplicate matrix entries");
        }
        if (general && r != c && std::abs(lower - upper) > 1e-12 * largest) {
            throw std::runtime_error("GENERAL input is not numerically symmetric");
        }
        const double v = symmetric || r == c ? lower + upper : 0.5 * lower + 0.5 * upper;
        if (v != 0.0) {
            ++matrix.ptr[c + 1];
            matrix.row.push_back(r);
            matrix.value.push_back(v);
        }
    }
    std::partial_sum(matrix.ptr.begin(), matrix.ptr.end(), matrix.ptr.begin());
    return matrix;
}

/**
 * Apply a symmetric matrix using its single stored triangle.
 *
 * Off-diagonal entries contribute to both rows. This operation also provides
 * an independent residual check that does not use the numerical factor.
 */
std::vector<double> Matrix::multiply(const std::vector<double>& x) const {
    // Validate the vector dimension before traversing the compressed columns
    if (x.size() != static_cast<std::size_t>(n)) {
        throw std::runtime_error("Matrix-vector dimension mismatch");
    }
    std::vector<double> result(n, 0.0);

    // Apply each stored lower entry and its symmetric counterpart
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = ptr[c]; p < ptr[c + 1]; ++p) {
            const int    r = row[p];
            const double v = value[p];
            result[r] += v * x[c];
            if (r != c) {
                result[c] += v * x[r];
            }
        }
    }
    return result;
}

/**
 * Form the symmetric permutation with order[new] = original.
 *
 * Each permuted coordinate is folded back into the lower triangle. Counting
 * columns first avoids sorting all matrix entries; only individual columns
 * need a local row sort after scattering.
 */
Matrix Matrix::permute(const std::vector<int>& order) const {
    // Build and validate the inverse permutation
    if (order.size() != static_cast<std::size_t>(n)) {
        throw std::runtime_error("Permutation dimension mismatch");
    }
    std::vector<int> inverse(n, -1);
    for (int i = 0; i < n; ++i) {
        if (order[i] < 0 || order[i] >= n || inverse[order[i]] != -1) {
            throw std::runtime_error("Invalid permutation");
        }
        inverse[order[i]] = i;
    }

    // Count lower-triangular columns in the new coordinate system
    Matrix result;
    result.n = n;
    result.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    result.row.resize(row.size());
    result.value.resize(value.size());
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = ptr[c]; p < ptr[c + 1]; ++p) {
            ++result.ptr[std::min(inverse[c], inverse[row[p]]) + 1];
        }
    }
    std::partial_sum(result.ptr.begin(), result.ptr.end(), result.ptr.begin());

    // Scatter coordinates, then restore sorted rows within each new column
    auto next = result.ptr;
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = ptr[c]; p < ptr[c + 1]; ++p) {
            const int         nc = std::min(inverse[c], inverse[row[p]]);
            const int         nr = std::max(inverse[c], inverse[row[p]]);
            const std::size_t q  = next[nc]++;
            result.row[q]   = nr;
            result.value[q] = value[p];
        }
    }
    std::vector<std::pair<int, double>> column;
    for (int c = 0; c < n; ++c) {
        column.clear();
        for (std::size_t p = result.ptr[c]; p < result.ptr[c + 1]; ++p) {
            column.emplace_back(result.row[p], result.value[p]);
        }
        std::sort(column.begin(), column.end());
        for (std::size_t k = 0; k < column.size(); ++k) {
            result.row[result.ptr[c] + k]   = column[k].first;
            result.value[result.ptr[c] + k] = column[k].second;
        }
    }
    return result;
}

} // namespace mf
