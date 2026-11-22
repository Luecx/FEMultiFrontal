/**
 * @file ldlt_schur.cpp
 * @brief Six-row, eight-column rank-k Schur updates with independent tiles.
 *
 * Unlike the SPD kernel, the two operands need not match: an indefinite
 * 2x2 D block is represented by L and L*D. Process entire output tiles
 * with shared right-hand panels, including holes from delayed pivots.
 */
#include "ldlt_schur.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf::dense {

void rank_k_ldlt_tiled(int origin, int dim, const std::vector<int>& active,
                       int rank, const double* left, const double* right,
                       double* packed) {
    if (rank == 0 || active.empty()) { return; }
    constexpr int MR = 6;
    constexpr int NR = 8;
    const int count = static_cast<int>(active.size());
    const int tile_count = (count + MR - 1) / MR;
    // OpenMP tasks must not throw across a task boundary. Workers record
    // nonfinite output; the caller throws after the implicit taskgroup joins.
    std::atomic<bool> nonfinite{false};

    auto update_tile = [&](int tile) {
        const int first = tile * MR;
        const int row_count = std::min(MR, count - first);
        double* target[MR] = {};
        int row[MR] = {};
        for (int r = 0; r < row_count; ++r) {
            row[r] = active[first+r];
            const std::size_t physical = static_cast<std::size_t>(origin + row[r]);
            target[r] = packed + physical * (physical + 1) / 2 + origin;
        }

        int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        // Rectangular tiles are strictly below the diagonal: all MR rows
        // contain all NR columns. Rows may be scattered, but each contiguous
        // column run can be loaded without gathers.
        for (; j + NR <= first; ) {
            const int col = active[j];
            if (active[j+NR-1] != col+NR-1) {
                // A gap makes this one column ineligible for the vector path.
                for (int r = 0; r < row_count; ++r) {
                    double v = target[r][col];
                    for (int p = 0; p < rank; ++p) {
                        const std::size_t offset = static_cast<std::size_t>(p) * dim;
                        v -= left[offset + row[r]] * right[offset + col];
                    }
                    if (!std::isfinite(v)) { nonfinite.store(true,std::memory_order_relaxed); return; }
                    target[r][col] = v;
                }
                ++j;
                continue;
            }
            __m256d acc0[MR], acc1[MR];
            for (int r = 0; r < row_count; ++r) {
                acc0[r] = _mm256_loadu_pd(target[r]+col);
                acc1[r] = _mm256_loadu_pd(target[r]+col+4);
            }
            for (int p = 0; p < rank; ++p) {
                const std::size_t offset = static_cast<std::size_t>(p) * dim;
                const __m256d u0 = _mm256_loadu_pd(right+offset+col);
                const __m256d u1 = _mm256_loadu_pd(right+offset+col+4);
                for (int r = 0; r < row_count; ++r) {
                    const __m256d multiplier = _mm256_set1_pd(left[offset+row[r]]);
                    acc0[r] = _mm256_fnmadd_pd(multiplier,u0,acc0[r]);
                    acc1[r] = _mm256_fnmadd_pd(multiplier,u1,acc1[r]);
                }
            }
            const __m256d limit = _mm256_set1_pd(std::numeric_limits<double>::max());
            const __m256d sign = _mm256_set1_pd(-0.0);
            for (int r = 0; r < row_count; ++r) {
                const __m256d check0 = _mm256_cmp_pd(_mm256_andnot_pd(sign,acc0[r]),limit,_CMP_LE_OQ);
                const __m256d check1 = _mm256_cmp_pd(_mm256_andnot_pd(sign,acc1[r]),limit,_CMP_LE_OQ);
                if (_mm256_movemask_pd(check0) != 15 || _mm256_movemask_pd(check1) != 15) {
                    nonfinite.store(true,std::memory_order_relaxed); return;
                }
                _mm256_storeu_pd(target[r]+col,acc0[r]);
                _mm256_storeu_pd(target[r]+col+4,acc1[r]);
            }
            j += NR;
        }
#endif
        // Remaining off-diagonal columns (including irregular index holes).
        // A single right coefficient is reused across the current output rows.
        for (; j < first; ++j) {
            const int col = active[j];
            double values[MR];
            for (int r = 0; r < row_count; ++r) { values[r] = target[r][col]; }
            for (int p = 0; p < rank; ++p) {
                const std::size_t offset = static_cast<std::size_t>(p)*dim;
                const double u = right[offset+col];
                for (int r = 0; r < row_count; ++r) {
                    values[r] -= left[offset+row[r]] * u;
                }
            }
            for (int r = 0; r < row_count; ++r) {
                if (!std::isfinite(values[r])) { nonfinite.store(true,std::memory_order_relaxed); return; }
                target[r][col] = values[r];
            }
        }

        // Triangular diagonal tile; never touch the upper triangle or inactive
        // entries, even if a pending pivot was removed from the active set.
        for (int r = 0; r < row_count; ++r) {
            for (int colpos = first; colpos <= first+r; ++colpos) {
                const int col = active[colpos];
                double v = target[r][col];
                for (int p = 0; p < rank; ++p) {
                    const std::size_t offset = static_cast<std::size_t>(p)*dim;
                    v -= left[offset+row[r]] * right[offset+col];
                }
                if (!std::isfinite(v)) { nonfinite.store(true,std::memory_order_relaxed); return; }
                target[r][col] = v;
            }
        }
    };

#ifdef _OPENMP
    // Update tiles own disjoint packed rows. Only use intra-front tasks when
    // enough work is available to amortize scheduling; small fronts stay serial.
    if (omp_in_parallel() && omp_get_num_threads() > 1 &&
        static_cast<double>(count)*count*rank >= 5e7 && tile_count >= 8) {
        const int jobs = std::min(tile_count,4*omp_get_num_threads());
#pragma omp taskloop grainsize(1) shared(update_tile)
        for (int job = 0; job < jobs; ++job) {
            const int first = static_cast<int>(tile_count*std::sqrt(static_cast<double>(job)/jobs));
            const int last = static_cast<int>(tile_count*std::sqrt(static_cast<double>(job+1)/jobs));
            for (int tile = first; tile < last; ++tile) { update_tile(tile); }
        }
        if (nonfinite.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Nonfinite LDLT Schur update");
        }
        return;
    }
#endif
    for (int tile = 0; tile < tile_count; ++tile) { update_tile(tile); }
    if (nonfinite.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Nonfinite LDLT Schur update");
    }
}

} // namespace mf::dense
