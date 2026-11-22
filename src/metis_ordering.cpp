/**
 * @file metis_ordering.cpp
 * @brief Optional scalar-graph METIS_NodeND ordering for comparison.
 *
 * Matrix stores only the lower triangular CSC entries. METIS expects an
 * undirected graph with both adjacency directions and no diagonal loops.
 * No user-visible or numerical path depends on METIS unless explicitly chosen.
 */
#include "metis_ordering.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef MF_WITH_METIS
#include <metis.h>
#endif

namespace mf {

std::vector<int> metis_nested_dissection(const Matrix& matrix) {
#ifndef MF_WITH_METIS
    (void)matrix;
    throw std::runtime_error(
        "--ordering metis requires an optional build: "
        "make WITH_METIS=1 (install the metis C library and headers)");
#else
    const int n = matrix.n;
    if (n < 0 || matrix.ptr.size() != static_cast<std::size_t>(n) + 1 ||
        matrix.row.size() != matrix.ptr.back()) {
        throw std::runtime_error("METIS ordering: invalid CSC input");
    }
    if (n < 2) {
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

    constexpr auto max_idx = std::numeric_limits<idx_t>::max();
    if (static_cast<std::uintmax_t>(n) > static_cast<std::uintmax_t>(max_idx)) {
        throw std::runtime_error("METIS ordering: vertex count exceeds idx_t");
    }

    // Convert strictly lower CSC into symmetric, zero-based adjacency.
    std::vector<idx_t> counts(n, 0);
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
            const int r = matrix.row[p];
            if (r < c || r >= n) {
                throw std::runtime_error("METIS ordering: expected lower CSC graph");
            }
            if (r != c) { ++counts[r]; ++counts[c]; }
        }
    }
    std::vector<idx_t> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        const std::uintmax_t next =
            static_cast<std::uintmax_t>(xadj[i]) +
            static_cast<std::uintmax_t>(counts[i]);
        if (next > static_cast<std::uintmax_t>(max_idx)) {
            throw std::runtime_error("METIS ordering: adjacency exceeds idx_t");
        }
        xadj[i + 1] = static_cast<idx_t>(next);
    }
    std::vector<idx_t> adjncy(static_cast<std::size_t>(xadj[n]));
    auto cursor = xadj;
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
            const int r = matrix.row[p];
            if (r != c) {
                adjncy[cursor[c]++] = static_cast<idx_t>(r);
                adjncy[cursor[r]++] = static_cast<idx_t>(c);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        std::sort(adjncy.begin() + xadj[i], adjncy.begin() + xadj[i + 1]);
    }

    idx_t nvtxs = static_cast<idx_t>(n);
    idx_t options[METIS_NOPTIONS];
    if (METIS_SetDefaultOptions(options) != METIS_OK) {
        throw std::runtime_error("METIS_SetDefaultOptions failed");
    }
    options[METIS_OPTION_NUMBERING] = 0;
    std::vector<idx_t> perm(n), iperm(n);
    const int status = METIS_NodeND(
        &nvtxs, xadj.data(), adjncy.data(), nullptr,
        options, perm.data(), iperm.data());
    if (status != METIS_OK) {
        throw std::runtime_error("METIS_NodeND failed with status " +
                                 std::to_string(status));
    }
    std::vector<int> order(n);
    std::vector<unsigned char> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        const auto old_id = perm[i];
        if (old_id < 0 || old_id >= n || seen[old_id] ||
            iperm[old_id] != i) {
            throw std::runtime_error("METIS_NodeND returned invalid permutation");
        }
        seen[old_id] = 1;
        order[i] = static_cast<int>(old_id);
    }
    return order;
#endif
}

} // namespace mf
