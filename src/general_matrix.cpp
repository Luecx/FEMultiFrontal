/**
 * @file general_matrix.cpp
 * @brief Imports and transforms full sparse operators for the pivoted solver.
 *
 * General numerical values remain independent across the diagonal. Graph
 * construction alone uses a symmetric structural union. Duplicate input
 * coordinates are summed before compression; normal equations are never formed.
 */
#include "general_matrix.h"

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
 * Read a complete real coordinate operator, preserving GENERAL asymmetry.
 *
 * SYMMETRIC files must supply one consistently oriented triangle, which is
 * expanded explicitly. Bounds and finite values are checked before duplicate
 * summation; cancellation zeros are then omitted from the full CSC matrix.
 */
GeneralMatrix GeneralMatrix::read_market(const std::string& path) {
    // Read the text once and validate its declared storage convention
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("Cannot open matrix: " + path); }
    const std::string input((std::istreambuf_iterator<char>(file)), {});
    const auto newline = input.find('\n');
    std::string header = input.substr(0, newline);
    if (!header.empty() && header.back() == '\r') { header.pop_back(); }
    const bool symmetric = header == "%%MatrixMarket matrix coordinate real symmetric";
    if (newline == std::string::npos || (!symmetric && header != "%%MatrixMarket matrix coordinate real general")) {
        throw std::runtime_error("Expected real coordinate GENERAL or SYMMETRIC Matrix Market input");
    }
    const char* cursor = input.data() + newline + 1;
    const char* end    = input.data() + input.size();
    auto skip = [&]() {
        while (cursor < end) {
            if (*cursor == '%') {
                while (cursor < end && *cursor != '\n') { ++cursor; }
            } else if (*cursor == ' ' || *cursor == '\r' || *cursor == '\n' || *cursor == '\t') {
                ++cursor;
            } else { break; }
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

    // Preserve orientation in full coordinates; mirror only explicitly symmetric input
    int rows = 0;
    int columns = 0;
    std::size_t count = 0;
    number(rows); number(columns); number(count);
    if (rows <= 0 || rows != columns || count > input.size() / 5) {
        throw std::runtime_error("Invalid square matrix dimensions or entry count");
    }
    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(count * (symmetric ? 2 : 1));
    int triangle = 0;
    for (std::size_t p = 0; p < count; ++p) {
        int r = 0;
        int c = 0;
        double value = 0.0;
        number(r); number(c); number(value);
        if (r < 1 || r > rows || c < 1 || c > columns || !std::isfinite(value)) {
            throw std::runtime_error("Invalid matrix coordinate or nonfinite value");
        }
        --r; --c;
        entries.emplace_back(c, r, value);
        if (symmetric && r != c) {
            const int side = r > c ? 1 : -1;
            if (triangle && triangle != side) {
                throw std::runtime_error("SYMMETRIC input must supply one consistently oriented triangle");
            }
            triangle = side;
            entries.emplace_back(r, c, value);
        }
    }
    skip();
    if (cursor != end) { throw std::runtime_error("Unexpected trailing Matrix Market entries"); }
    std::sort(entries.begin(), entries.end());

    // Sum duplicates in each actual coordinate, without comparing opposite entries
    GeneralMatrix matrix;
    matrix.n = rows;
    matrix.ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    matrix.row.reserve(entries.size());
    matrix.value.reserve(entries.size());
    for (std::size_t p = 0; p < entries.size();) {
        const int c = std::get<0>(entries[p]);
        const int r = std::get<1>(entries[p]);
        double value = 0.0;
        do { value += std::get<2>(entries[p++]); }
        while (p < entries.size() && std::get<0>(entries[p]) == c && std::get<1>(entries[p]) == r);
        if (!std::isfinite(value)) { throw std::runtime_error("Nonfinite duplicate-entry sum"); }
        if (value != 0.0) {
            ++matrix.ptr[c + 1];
            matrix.row.push_back(r);
            matrix.value.push_back(value);
        }
    }
    std::partial_sum(matrix.ptr.begin(), matrix.ptr.end(), matrix.ptr.begin());
    return matrix;
}

/**
 * Expand one canonical lower triangle into both halves without changing values.
 *
 * Column counts allocate exact storage. The source's sorted lower rows and
 * ascending source-column traversal produce sorted complete output columns.
 */
GeneralMatrix GeneralMatrix::from_symmetric(const Matrix& matrix) {
    // Count every original entry and the transpose of each off-diagonal entry
    GeneralMatrix result;
    result.n = matrix.n;
    result.ptr.assign(static_cast<std::size_t>(matrix.n) + 1, 0);
    for (int c = 0; c < matrix.n; ++c) {
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
            ++result.ptr[c + 1];
            if (matrix.row[p] != c) { ++result.ptr[matrix.row[p] + 1]; }
        }
    }
    std::partial_sum(result.ptr.begin(), result.ptr.end(), result.ptr.begin());
    result.row.resize(result.ptr.back());
    result.value.resize(result.ptr.back());
    auto next = result.ptr;

    // Insert both actual coordinates, including the diagonal only once
    for (int c = 0; c < matrix.n; ++c) {
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
            const int r = matrix.row[p];
            const auto q = next[c]++;
            result.row[q] = r;
            result.value[q] = matrix.value[p];
            if (r != c) {
                const auto t = next[r]++;
                result.row[t] = c;
                result.value[t] = matrix.value[p];
            }
        }
    }
    return result;
}

/** Apply each stored full entry exactly once to its equation row. */
std::vector<double> GeneralMatrix::multiply(const std::vector<double>& x) const {
    // Check dimensions before applying the full nonsymmetric operator
    if (x.size() != static_cast<std::size_t>(n)) { throw std::runtime_error("Matrix-vector dimension mismatch"); }
    std::vector<double> result(n, 0.0);
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = ptr[c]; p < ptr[c + 1]; ++p) { result[row[p]] += value[p] * x[c]; }
    }
    return result;
}

/**
 * Form an independently row- and column-permuted full operator.
 *
 * Both supplied vectors map new coordinates to original coordinates. The
 * inverse row map remaps entries; each resulting column is sorted afterward.
 */
GeneralMatrix GeneralMatrix::permute(const std::vector<int>& rows, const std::vector<int>& columns) const {
    // Validate two independent bijections, including equation and variable spaces
    if (rows.size() != static_cast<std::size_t>(n) || columns.size() != static_cast<std::size_t>(n)) {
        throw std::runtime_error("General permutation dimension mismatch");
    }
    std::vector<int> inverse_rows(n, -1);
    std::vector<int> inverse_columns(n, -1);
    for (int i = 0; i < n; ++i) {
        if (rows[i] < 0 || rows[i] >= n || columns[i] < 0 || columns[i] >= n ||
            inverse_rows[rows[i]] != -1 || inverse_columns[columns[i]] != -1) {
            throw std::runtime_error("Invalid general matrix permutation");
        }
        inverse_rows[rows[i]] = i;
        inverse_columns[columns[i]] = i;
    }

    // Rebuild columns without folding or mirroring numerical coordinates
    GeneralMatrix result;
    result.n = n;
    result.ptr.push_back(0);
    std::vector<std::pair<int, double>> column;
    for (int c = 0; c < n; ++c) {
        column.clear();
        for (std::size_t p = ptr[columns[c]]; p < ptr[columns[c] + 1]; ++p) {
            column.emplace_back(inverse_rows[row[p]], value[p]);
        }
        std::sort(column.begin(), column.end());
        for (const auto& [r, v] : column) { result.row.push_back(r); result.value.push_back(v); }
        result.ptr.push_back(result.row.size());
    }
    return result;
}

/**
 * Construct the symmetric structural union for ordering and a frontal tree.
 *
 * Unit-valued graph entries describe adjacency only. The numerical A remains
 * unchanged; neither A+A^T values nor normal equations are used in a solve.
 */
Matrix GeneralMatrix::ordering_graph() const {
    // Fold coordinates structurally, deduplicating opposite numerical entries
    std::vector<std::pair<int, int>> edges;
    edges.reserve(row.size());
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = ptr[c]; p < ptr[c + 1]; ++p) {
            edges.emplace_back(std::min(c, row[p]), std::max(c, row[p]));
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    Matrix result;
    result.n = n;
    result.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& [c, r] : edges) { ++result.ptr[c + 1]; result.row.push_back(r); }
    std::partial_sum(result.ptr.begin(), result.ptr.end(), result.ptr.begin());
    result.value.assign(result.row.size(), 1.0);
    return result;
}

} // namespace mf
