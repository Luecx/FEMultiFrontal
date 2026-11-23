/**
 * @file ldlt_tiled_front.cpp
 * @brief Direct Schur updates on optional 32x32 tile-major frontal storage.
 *
 * Unlike staging a conventional row-packed triangle for every flush, this
 * kernel keeps the output in the same tile-major storage throughout the front's
 * lifetime. Panel factors are packed to the physical tile coordinates once per
 * flush, with zeros for inactive rows and columns. Only active lower entries
 * are read/written, including when delayed pivots have left index holes.
 */
#include "ldlt_tiled_front.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {

void rank_k_ldlt_tiled_front(int origin, int dim, const std::vector<int>& active,
                             int rank, const double* left, const double* right,
                             double* packed) {
    if (rank <= 0 || active.empty()) { return; }

    constexpr int BS = front_tile_size;
    const int tiles = (origin + active.back()) / BS + 1;
    std::vector<std::vector<int>> selected(static_cast<std::size_t>(tiles));
    const std::size_t panel_size = static_cast<std::size_t>(rank) * BS;
    std::vector<double> a(static_cast<std::size_t>(tiles) * panel_size, 0.0);
    std::vector<double> b(static_cast<std::size_t>(tiles) * panel_size, 0.0);

    for (int r : active) {
        const int physical = origin + r;
        const int tile = physical / BS;
        const int lane = physical % BS;
        selected[tile].push_back(lane);
        for (int p = 0; p < rank; ++p) {
            const std::size_t base = static_cast<std::size_t>(tile) * panel_size +
                                     static_cast<std::size_t>(p) * BS + lane;
            a[base] = left[static_cast<std::size_t>(p)*dim + r];
            b[base] = right[static_cast<std::size_t>(p)*dim + r];
        }
    }

    // Each lower output tile is traversed once and remains physically dense.
    // The row/column-lane lists exclude all delayed-pivot holes.
    for (int ti = 0; ti < tiles; ++ti) {
        const auto& rows = selected[ti];
        if (rows.empty()) { continue; }
        const double* const ap = a.data() + static_cast<std::size_t>(ti)*panel_size;
        for (int tj = 0; tj <= ti; ++tj) {
            const auto& cols = selected[tj];
            if (cols.empty()) { continue; }
            const double* const bp = b.data() + static_cast<std::size_t>(tj)*panel_size;
            const std::size_t tile_index =
                static_cast<std::size_t>(ti)*(ti+1)/2 + tj;
            double* const tile = packed + tile_index*front_tile_entries;

            for (std::size_t ir = 0; ir < rows.size(); ir += 4) {
                const int nr = static_cast<int>(std::min<std::size_t>(4,rows.size()-ir));
                std::size_t jc = 0;
#if defined(__AVX2__) && defined(__FMA__)
                // Four rows by eight contiguous physical columns: retain the
                // successful AVX2 register tile but eliminate packed-C
                // row-stride jumps between logical matrix tiles.
                if (nr == 4) {
                    for (; jc + 8 <= cols.size(); jc += 8) {
                        const int first_col = cols[jc];
                        if (cols[jc+7] != first_col+7 ||
                            (ti == tj && first_col+7 > rows[ir])) {
                            break;
                        }
                        __m256d c0[4], c1[4];
                        for (int r = 0; r < 4; ++r) {
                            const double* row = tile + static_cast<std::size_t>(rows[ir+r])*BS
                                                + first_col;
                            c0[r] = _mm256_loadu_pd(row);
                            c1[r] = _mm256_loadu_pd(row+4);
                        }
                        for (int p = 0; p < rank; ++p) {
                            const double* const right_p = bp + static_cast<std::size_t>(p)*BS
                                                          + first_col;
                            const __m256d v0 = _mm256_loadu_pd(right_p);
                            const __m256d v1 = _mm256_loadu_pd(right_p+4);
                            for (int r = 0; r < 4; ++r) {
                                const __m256d left_r = _mm256_set1_pd(
                                    ap[static_cast<std::size_t>(p)*BS + rows[ir+r]]);
                                c0[r] = _mm256_fnmadd_pd(left_r, v0, c0[r]);
                                c1[r] = _mm256_fnmadd_pd(left_r, v1, c1[r]);
                            }
                        }
                        for (int r = 0; r < 4; ++r) {
                            alignas(32) double check[8];
                            _mm256_store_pd(check,c0[r]);
                            _mm256_store_pd(check+4,c1[r]);
                            for (double value : check) {
                                if (!std::isfinite(value)) {
                                    throw std::runtime_error(
                                        "Nonfinite tile-packed LDLT Schur update");
                                }
                            }
                            double* row = tile + static_cast<std::size_t>(rows[ir+r])*BS
                                          + first_col;
                            _mm256_storeu_pd(row,c0[r]);
                            _mm256_storeu_pd(row+4,c1[r]);
                        }
                    }
                }
#endif
                // Triangular diagonal tiles, irregular index sets and vector
                // tails use the same arithmetic in scalar pivot order.
                for (int r = 0; r < nr; ++r) {
                    const int row_lane = rows[ir+r];
                    double* const row = tile + static_cast<std::size_t>(row_lane)*BS;
                    for (std::size_t j = jc; j < cols.size(); ++j) {
                        const int col_lane = cols[j];
                        if (ti == tj && col_lane > row_lane) { break; }
                        double value = row[col_lane];
                        for (int p = 0; p < rank; ++p) {
                            const std::size_t base = static_cast<std::size_t>(p)*BS;
                            value -= ap[base+row_lane] * bp[base+col_lane];
                        }
                        if (!std::isfinite(value)) {
                            throw std::runtime_error(
                                "Nonfinite tile-packed LDLT Schur update");
                        }
                        row[col_lane] = value;
                    }
                }
            }
        }
    }
}

} // namespace mf::dense
