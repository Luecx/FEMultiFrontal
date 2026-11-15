/**
 * @file dense_tiled.cpp
 * @brief Updates blocked temporary Schur triangles with native AVX2/FMA kernels.
 *
 * Six-row/eight-row packed operands feed native register kernels. Output work stays
 * within a 32-by-32 tile, with 6/4/2/1-by-8 register kernels and exact scalar boundary
 * handling. Existing OpenMP teams may execute disjoint tile-row bands; no BLAS
 * library or extra parallel region is introduced.
 */
#include "dense_tiled.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf::dense {

namespace {

/**
 * Update Rows by eight entries within one dense output tile.
 *
 * A uses six-row rank-major packing; B uses eight-row packing. Specializations
 * for 1/2/4/6 rows cover partial tile boundaries without scalar dot products.
 * Each output retains one accumulator per lane and ascending rank order.
 */
template<int Rows>
void subtract_rows(int k, double* target, const double* a, const double* b) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d low[Rows];
    __m256d high[Rows];
    for (int r = 0; r < Rows; ++r) {
        low[r]  = _mm256_loadu_pd(target + r * schur_tile_size);
        high[r] = _mm256_loadu_pd(target + r * schur_tile_size + 4);
    }
    for (int p = 0; p < k; ++p) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        for (int r = 0; r < Rows; ++r) {
            const __m256d scale = _mm256_set1_pd(a[r]);
            low[r]  = _mm256_fnmadd_pd(scale, b0, low[r]);
            high[r] = _mm256_fnmadd_pd(scale, b1, high[r]);
        }
        a += 6;
        b += 8;
    }
    for (int r = 0; r < Rows; ++r) {
        _mm256_storeu_pd(target + r * schur_tile_size, low[r]);
        _mm256_storeu_pd(target + r * schur_tile_size + 4, high[r]);
    }
#endif
}

} // namespace


/**
 * Apply a rank-k update to the active suffix of a padded tiled lower triangle.
 *
 * The packed columns describe n active variables. Physical indices add
 * origin+1 because the contribution omits the original first pivot. Inherited
 * buffers keep their original tile coordinates, including partial leading
 * tiles. Only active lower entries are modified; eliminated and padded entries
 * remain untouched. Each output entry retains ascending rank accumulation.
 */
void rank_k_subtract_tiled(int n, int k, const double* const* columns, double* tiled,
                           int origin, std::vector<double>& scratch) {
    if (n <= 0 || k <= 0) { return; }
    const int first       = origin + 1;
    const int last        = first + n;
    const int first_block = first / schur_tile_size;
    const int blocks      = (last - 1) / schur_tile_size - first_block + 1;

    // Pack six-row A and eight-row B panels once for the active suffix
#if defined(__AVX2__) && defined(__FMA__)
    const std::size_t a_stride = static_cast<std::size_t>(k) * 6;
    const std::size_t b_stride = static_cast<std::size_t>(k) * 8;
    const std::size_t a_size   = (((static_cast<std::size_t>(n) + 5) / 6 * a_stride) + 7) / 8 * 8;
    scratch.resize(a_size + (static_cast<std::size_t>(n) + 7) / 8 * b_stride + 8);
    const auto address = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* a_data = scratch.data() + ((64 - (address & 63)) & 63) / sizeof(double);
    double* b_data = a_data + a_size;
    for (int base = 0; base < n; base += 6) {
        const int count = std::min(6, n - base);
        for (int p = 0; p < k; ++p) {
            std::copy_n(columns[p] + base, count, a_data + static_cast<std::size_t>(base / 6) * a_stride + p * 6);
        }
    }
    for (int base = 0; base < n; base += 8) {
        const int count = std::min(8, n - base);
        for (int p = 0; p < k; ++p) {
            std::copy_n(columns[p] + base, count, b_data + static_cast<std::size_t>(base / 8) * b_stride + p * 8);
        }
    }
#else
    (void)scratch;
#endif

    // Tile rows own disjoint output entries, including partial diagonal tiles
    auto update_row = [&](int block) {
        const int block_row = first_block + block;
        const int row_first = std::max(first, block_row * schur_tile_size);
        const int row_last  = std::min(last, (block_row + 1) * schur_tile_size);
        for (int block_col = first_block; block_col <= block_row; ++block_col) {
            const int col_first = std::max(first, block_col * schur_tile_size);
            const int col_last  = std::min(last, (block_col + 1) * schur_tile_size);
            int r = row_first;
            for (; r < row_last;) {
                const int i = r - first;
                int c = col_first;
#if defined(__AVX2__) && defined(__FMA__)
                const int available = std::min(6 - i % 6, row_last - r);
                const int rows = available == 6 ? 6 : available >= 4 ? 4 : available >= 2 ? 2 : 1;
                // Leading partial columns cannot cross an eight-row input panel
                for (; c < col_last && (c - first) % 8 != 0; ++c) {
                    for (int rr = r; rr < r + rows; ++rr) {
                        if (c > rr) { continue; }
                        double& target = tiled[tiled_lower_offset(rr, c)];
                        for (int p = 0; p < k; ++p) { target -= columns[p][rr-first] * columns[p][c-first]; }
                    }
                }
                for (; c + 8 <= col_last && c + 7 <= r; c += 8) {
                    double* target = tiled + tiled_lower_offset(r, c);
                    const double* a = a_data + static_cast<std::size_t>(i / 6) * a_stride + i % 6;
                    const double* b = b_data + static_cast<std::size_t>((c-first) / 8) * b_stride;
                    if (rows == 6) { subtract_rows<6>(k, target, a, b); }
                    else if (rows == 4) { subtract_rows<4>(k, target, a, b); }
                    else if (rows == 2) { subtract_rows<2>(k, target, a, b); }
                    else { subtract_rows<1>(k, target, a, b); }
                }
                // Complete the diagonal strip and partial final column tile
                for (int rr = r; rr < r + rows; ++rr) {
                    for (int cc = c; cc < col_last && cc <= rr; ++cc) {
                        double& target = tiled[tiled_lower_offset(rr, cc)];
                        for (int p = 0; p < k; ++p) { target -= columns[p][rr-first] * columns[p][cc-first]; }
                    }
                }
                r += rows;
                continue;
#endif
                // Portable implementation and incomplete output-row groups
                for (; c < col_last && c <= r; ++c) {
                    double& target = tiled[tiled_lower_offset(r, c)];
                    for (int p = 0; p < k; ++p) { target -= columns[p][i] * columns[p][c-first]; }
                }
                ++r;
            }
        }
    };

    // Match the existing work threshold so this experiment isolates storage
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads() > 1 && static_cast<double>(n) * n * k >= 5e7) {
        const int jobs = std::min(blocks, 4 * omp_get_num_threads());
#pragma omp taskloop grainsize(1) shared(update_row)
        for (int job = 0; job < jobs; ++job) {
            const int begin = static_cast<int>(blocks * std::sqrt(static_cast<double>(job) / jobs));
            const int end   = static_cast<int>(blocks * std::sqrt(static_cast<double>(job + 1) / jobs));
            for (int block = begin; block < end; ++block) { update_row(block); }
        }
    } else
#endif
    {
        for (int block = 0; block < blocks; ++block) { update_row(block); }
    }
}

} // namespace mf::dense
