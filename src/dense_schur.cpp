/**
 * @file dense_schur.cpp
 * @brief Implements alternative raw AVX2 Schur kernels for scalar multifrontal updates.
 *
 * All kernels subtract the same sum of outer products in row-packed lower
 * storage. The wide kernel increases independent FMA chains; the packed kernels
 * reorganize factor suffixes into aligned panels before updating the front.
 * Packing is performed on every call using caller-owned reusable scratch.
 * Neither variant changes pivots, fronts or the numerical factor representation.
 */
#include "dense.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {

/**
 * Subtract X X^T using a direct 4-by-12 AVX2/FMA register tile.
 *
 * Twelve independent vector accumulators cover more FMA latency than the
 * baseline eight. One broadcast register is reused row by row to limit register
 * pressure. Short calls and non-AVX builds retain the baseline implementation.
 */
void rank_k_subtract_wide(int n, int k, const double* const* columns, double* packed, int origin) {
    // Keep the established small-rank path and portable implementation
#if !defined(__AVX2__) || !defined(__FMA__)
    rank_k_subtract(n, k, columns, packed, origin);
#else
    if (k <= 1) {
        rank_k_subtract(n, k, columns, packed, origin);
        return;
    }

    // Keep each strictly lower rectangular output tile in twelve AVX accumulators
    for (int i = 0; i < n; i += 4) {
        int j = 0;
        if (i + 4 <= n) {
            double* target[4];
            for (int r = 0; r < 4; ++r) {
                const std::size_t row = static_cast<std::size_t>(origin + i + r);
                target[r] = packed + row * (row + 1) / 2 + origin;
            }
            for (; j + 12 <= i; j += 12) {
                __m256d c00 = _mm256_loadu_pd(target[0] + j + 0);
                __m256d c01 = _mm256_loadu_pd(target[0] + j + 4);
                __m256d c02 = _mm256_loadu_pd(target[0] + j + 8);
                __m256d c10 = _mm256_loadu_pd(target[1] + j + 0);
                __m256d c11 = _mm256_loadu_pd(target[1] + j + 4);
                __m256d c12 = _mm256_loadu_pd(target[1] + j + 8);
                __m256d c20 = _mm256_loadu_pd(target[2] + j + 0);
                __m256d c21 = _mm256_loadu_pd(target[2] + j + 4);
                __m256d c22 = _mm256_loadu_pd(target[2] + j + 8);
                __m256d c30 = _mm256_loadu_pd(target[3] + j + 0);
                __m256d c31 = _mm256_loadu_pd(target[3] + j + 4);
                __m256d c32 = _mm256_loadu_pd(target[3] + j + 8);
                for (int p = 0; p < k; ++p) {
                    const double* x = columns[p];
                    const __m256d b0 = _mm256_loadu_pd(x + j + 0);
                    const __m256d b1 = _mm256_loadu_pd(x + j + 4);
                    const __m256d b2 = _mm256_loadu_pd(x + j + 8);
                    __m256d a;
                    a = _mm256_set1_pd(x[i + 0]);
                    c00 = _mm256_fnmadd_pd(a, b0, c00);
                    c01 = _mm256_fnmadd_pd(a, b1, c01);
                    c02 = _mm256_fnmadd_pd(a, b2, c02);
                    a = _mm256_set1_pd(x[i + 1]);
                    c10 = _mm256_fnmadd_pd(a, b0, c10);
                    c11 = _mm256_fnmadd_pd(a, b1, c11);
                    c12 = _mm256_fnmadd_pd(a, b2, c12);
                    a = _mm256_set1_pd(x[i + 2]);
                    c20 = _mm256_fnmadd_pd(a, b0, c20);
                    c21 = _mm256_fnmadd_pd(a, b1, c21);
                    c22 = _mm256_fnmadd_pd(a, b2, c22);
                    a = _mm256_set1_pd(x[i + 3]);
                    c30 = _mm256_fnmadd_pd(a, b0, c30);
                    c31 = _mm256_fnmadd_pd(a, b1, c31);
                    c32 = _mm256_fnmadd_pd(a, b2, c32);
                }
                _mm256_storeu_pd(target[0] + j + 0, c00);
                _mm256_storeu_pd(target[0] + j + 4, c01);
                _mm256_storeu_pd(target[0] + j + 8, c02);
                _mm256_storeu_pd(target[1] + j + 0, c10);
                _mm256_storeu_pd(target[1] + j + 4, c11);
                _mm256_storeu_pd(target[1] + j + 8, c12);
                _mm256_storeu_pd(target[2] + j + 0, c20);
                _mm256_storeu_pd(target[2] + j + 4, c21);
                _mm256_storeu_pd(target[2] + j + 8, c22);
                _mm256_storeu_pd(target[3] + j + 0, c30);
                _mm256_storeu_pd(target[3] + j + 4, c31);
                _mm256_storeu_pd(target[3] + j + 8, c32);
            }
        }

        // Complete the diagonal band and partial rows without out-of-range loads
        for (int r = i; r < i + 4 && r < n; ++r) {
            const std::size_t row = static_cast<std::size_t>(origin + r);
            double* target = packed + row * (row + 1) / 2 + origin;
            for (int c = j; c <= r; ++c) {
                double value = target[c];
                for (int p = 0; p < k; ++p) { value -= columns[p][r] * columns[p][c]; }
                target[c] = value;
            }
        }
    }
#endif
}

/**
 * Pack the input suffixes into eight-row panels and subtract X X^T.
 *
 * A panel stores k consecutive eight-entry slices, one per factor column.
 * The inner rank loop therefore reads two short linear streams instead of
 * revisiting k widely separated factor columns for every output tile. Scratch
 * is resized only as necessary, but all panel copies are included in this call.
 * The scalar remainder uses the original suffixes and preserves update order.
 */
void rank_k_subtract_packed(
    int                  n,
    int                  k,
    const double* const* columns,
    double*              packed,
    int                  origin,
    std::vector<double>& scratch
) {
    // Avoid panel construction when arithmetic is too small to amortize the copy
#if !defined(__AVX2__) || !defined(__FMA__)
    (void)scratch;
    rank_k_subtract(n, k, columns, packed, origin);
#else
    if (n < 64 || k <= 1) {
        rank_k_subtract(n, k, columns, packed, origin);
        return;
    }

    // Copy each factor suffix into contiguous rank-major eight-row panels
    const std::size_t panels = (static_cast<std::size_t>(n) + 7) / 8;
    const std::size_t stride = static_cast<std::size_t>(k) * 8;
    scratch.resize(panels * stride + 8);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* panel_data = scratch.data() + ((64 - (address & 63)) & 63) / sizeof(double);
    for (std::size_t panel = 0; panel < panels; ++panel) {
        const int base  = static_cast<int>(panel * 8);
        const int count = std::min(8, n - base);
        for (int p = 0; p < k; ++p) {
            double* target = panel_data + panel * stride + p * 8;
            if (count == 8) {
                _mm256_store_pd(target, _mm256_loadu_pd(columns[p] + base));
                _mm256_store_pd(target + 4, _mm256_loadu_pd(columns[p] + base + 4));
            } else {
                std::copy_n(columns[p] + base, count, target);
            }
        }
    }

    // Reuse the same packed A panel across all B panels to the left of the diagonal
    for (int i = 0; i < n; i += 4) {
        int j = 0;
        if (i + 4 <= n) {
            double* target[4];
            for (int r = 0; r < 4; ++r) {
                const std::size_t row = static_cast<std::size_t>(origin + i + r);
                target[r] = packed + row * (row + 1) / 2 + origin;
            }
            const double* a_panel = panel_data + static_cast<std::size_t>(i / 8) * stride + i % 8;
            for (; j + 8 <= i; j += 8) {
                __m256d c00 = _mm256_loadu_pd(target[0] + j + 0);
                __m256d c01 = _mm256_loadu_pd(target[0] + j + 4);
                __m256d c10 = _mm256_loadu_pd(target[1] + j + 0);
                __m256d c11 = _mm256_loadu_pd(target[1] + j + 4);
                __m256d c20 = _mm256_loadu_pd(target[2] + j + 0);
                __m256d c21 = _mm256_loadu_pd(target[2] + j + 4);
                __m256d c30 = _mm256_loadu_pd(target[3] + j + 0);
                __m256d c31 = _mm256_loadu_pd(target[3] + j + 4);
                const double* a = a_panel;
                const double* b = panel_data + static_cast<std::size_t>(j / 8) * stride;
                for (int p = 0; p < k; ++p) {
                    const __m256d b0 = _mm256_load_pd(b);
                    const __m256d b1 = _mm256_load_pd(b + 4);
                    __m256d scale;
                    scale = _mm256_set1_pd(a[0]);
                    c00 = _mm256_fnmadd_pd(scale, b0, c00);
                    c01 = _mm256_fnmadd_pd(scale, b1, c01);
                    scale = _mm256_set1_pd(a[1]);
                    c10 = _mm256_fnmadd_pd(scale, b0, c10);
                    c11 = _mm256_fnmadd_pd(scale, b1, c11);
                    scale = _mm256_set1_pd(a[2]);
                    c20 = _mm256_fnmadd_pd(scale, b0, c20);
                    c21 = _mm256_fnmadd_pd(scale, b1, c21);
                    scale = _mm256_set1_pd(a[3]);
                    c30 = _mm256_fnmadd_pd(scale, b0, c30);
                    c31 = _mm256_fnmadd_pd(scale, b1, c31);
                    a += 8;
                    b += 8;
                }
                _mm256_storeu_pd(target[0] + j + 0, c00);
                _mm256_storeu_pd(target[0] + j + 4, c01);
                _mm256_storeu_pd(target[1] + j + 0, c10);
                _mm256_storeu_pd(target[1] + j + 4, c11);
                _mm256_storeu_pd(target[2] + j + 0, c20);
                _mm256_storeu_pd(target[2] + j + 4, c21);
                _mm256_storeu_pd(target[3] + j + 0, c30);
                _mm256_storeu_pd(target[3] + j + 4, c31);
            }
        }

        // Original suffixes cover the diagonal and final incomplete panel exactly
        for (int r = i; r < i + 4 && r < n; ++r) {
            const std::size_t row = static_cast<std::size_t>(origin + r);
            double* target = packed + row * (row + 1) / 2 + origin;
            for (int c = j; c <= r; ++c) {
                double value = target[c];
                for (int p = 0; p < k; ++p) { value -= columns[p][r] * columns[p][c]; }
                target[c] = value;
            }
        }
    }
#endif
}

/**
 * Use separately packed six-row and eight-row panels for a 6-by-8 FMA tile.
 *
 * Twelve independent accumulator chains cover FMA latency while leaving room
 * for two B vectors and one broadcast in AVX2's sixteen registers. B panels
 * start at cache-line boundaries; A panels contain six scalars per rank step.
 * Both packing passes are included and reuse the caller's scratch allocation.
 */
void rank_k_subtract_packed6(
    int                  n,
    int                  k,
    const double* const* columns,
    double*              packed,
    int                  origin,
    std::vector<double>& scratch
) {
    // Preserve short calls and the portable implementation without panel construction
#if !defined(__AVX2__) || !defined(__FMA__)
    (void)scratch;
    rank_k_subtract(n, k, columns, packed, origin);
#else
    if (n < 64 || k <= 1) {
        rank_k_subtract(n, k, columns, packed, origin);
        return;
    }

    // Reserve one aligned workspace for both panel layouts, rounding the B start to 64 bytes
    const std::size_t a_count  = (static_cast<std::size_t>(n) + 5) / 6;
    const std::size_t b_count  = (static_cast<std::size_t>(n) + 7) / 8;
    const std::size_t a_stride = static_cast<std::size_t>(k) * 6;
    const std::size_t b_stride = static_cast<std::size_t>(k) * 8;
    const std::size_t a_size   = (a_count * a_stride + 7) / 8 * 8;
    scratch.resize(a_size + b_count * b_stride + 8);
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* a_data = scratch.data() + ((64 - (address & 63)) & 63) / sizeof(double);
    double* b_data = a_data + a_size;

    // Pack A and B in the exact rank-major layout consumed by the microkernel
    for (std::size_t panel = 0; panel < a_count; ++panel) {
        const int base  = static_cast<int>(panel * 6);
        const int count = std::min(6, n - base);
        for (int p = 0; p < k; ++p) {
            std::copy_n(columns[p] + base, count, a_data + panel * a_stride + p * 6);
        }
    }
    for (std::size_t panel = 0; panel < b_count; ++panel) {
        const int base  = static_cast<int>(panel * 8);
        const int count = std::min(8, n - base);
        for (int p = 0; p < k; ++p) {
            double* target = b_data + panel * b_stride + p * 8;
            if (count == 8) {
                _mm256_store_pd(target, _mm256_loadu_pd(columns[p] + base));
                _mm256_store_pd(target + 4, _mm256_loadu_pd(columns[p] + base + 4));
            } else {
                std::copy_n(columns[p] + base, count, target);
            }
        }
    }

    // Accumulate six packed output rows with twelve independent vector FMA chains
    auto update_rows = [&](int tile) {
        const int i = tile * 6;
        int j = 0;
        if (i + 6 <= n) {
            double* target[6];
            for (int r = 0; r < 6; ++r) {
                const std::size_t row = static_cast<std::size_t>(origin + i + r);
                target[r] = packed + row * (row + 1) / 2 + origin;
            }
            const double* a_panel = a_data + static_cast<std::size_t>(i / 6) * a_stride;
            for (; j + 8 <= i; j += 8) {
                __m256d c00 = _mm256_loadu_pd(target[0] + j + 0);
                __m256d c01 = _mm256_loadu_pd(target[0] + j + 4);
                __m256d c10 = _mm256_loadu_pd(target[1] + j + 0);
                __m256d c11 = _mm256_loadu_pd(target[1] + j + 4);
                __m256d c20 = _mm256_loadu_pd(target[2] + j + 0);
                __m256d c21 = _mm256_loadu_pd(target[2] + j + 4);
                __m256d c30 = _mm256_loadu_pd(target[3] + j + 0);
                __m256d c31 = _mm256_loadu_pd(target[3] + j + 4);
                __m256d c40 = _mm256_loadu_pd(target[4] + j + 0);
                __m256d c41 = _mm256_loadu_pd(target[4] + j + 4);
                __m256d c50 = _mm256_loadu_pd(target[5] + j + 0);
                __m256d c51 = _mm256_loadu_pd(target[5] + j + 4);
                const double* a = a_panel;
                const double* b = b_data + static_cast<std::size_t>(j / 8) * b_stride;
                for (int p = 0; p < k; ++p) {
                    const __m256d b0 = _mm256_load_pd(b);
                    const __m256d b1 = _mm256_load_pd(b + 4);
                    __m256d scale;
                    scale = _mm256_set1_pd(a[0]);
                    c00 = _mm256_fnmadd_pd(scale, b0, c00);
                    c01 = _mm256_fnmadd_pd(scale, b1, c01);
                    scale = _mm256_set1_pd(a[1]);
                    c10 = _mm256_fnmadd_pd(scale, b0, c10);
                    c11 = _mm256_fnmadd_pd(scale, b1, c11);
                    scale = _mm256_set1_pd(a[2]);
                    c20 = _mm256_fnmadd_pd(scale, b0, c20);
                    c21 = _mm256_fnmadd_pd(scale, b1, c21);
                    scale = _mm256_set1_pd(a[3]);
                    c30 = _mm256_fnmadd_pd(scale, b0, c30);
                    c31 = _mm256_fnmadd_pd(scale, b1, c31);
                    scale = _mm256_set1_pd(a[4]);
                    c40 = _mm256_fnmadd_pd(scale, b0, c40);
                    c41 = _mm256_fnmadd_pd(scale, b1, c41);
                    scale = _mm256_set1_pd(a[5]);
                    c50 = _mm256_fnmadd_pd(scale, b0, c50);
                    c51 = _mm256_fnmadd_pd(scale, b1, c51);
                    a += 6;
                    b += 8;
                }
                _mm256_storeu_pd(target[0] + j + 0, c00);
                _mm256_storeu_pd(target[0] + j + 4, c01);
                _mm256_storeu_pd(target[1] + j + 0, c10);
                _mm256_storeu_pd(target[1] + j + 4, c11);
                _mm256_storeu_pd(target[2] + j + 0, c20);
                _mm256_storeu_pd(target[2] + j + 4, c21);
                _mm256_storeu_pd(target[3] + j + 0, c30);
                _mm256_storeu_pd(target[3] + j + 4, c31);
                _mm256_storeu_pd(target[4] + j + 0, c40);
                _mm256_storeu_pd(target[4] + j + 4, c41);
                _mm256_storeu_pd(target[5] + j + 0, c50);
                _mm256_storeu_pd(target[5] + j + 4, c51);
            }
        }

        // Complete diagonal bands and incomplete row panels using the original suffixes
        for (int r = i; r < i + 6 && r < n; ++r) {
            const std::size_t row = static_cast<std::size_t>(origin + r);
            double* target = packed + row * (row + 1) / 2 + origin;
            for (int c = j; c <= r; ++c) {
                double value = target[c];
                for (int p = 0; p < k; ++p) { value -= columns[p][r] * columns[p][c]; }
                target[c] = value;
            }
        }
    };

    // Use the factorization's existing team. The implicit taskgroup retains the
    // panel workspace until every disjoint row tile has completed.
    const int tiles = (n + 5) / 6;
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads() > 1 && static_cast<double>(n) * n * k >= 5e7) {
        // Row work grows linearly down the triangle. Square-root boundaries
        // assign approximately equal areas instead of equal row counts, while
        // keeping each task large enough to amortize scheduling overhead.
        const int jobs = std::min(tiles, 4 * omp_get_num_threads());
#pragma omp taskloop grainsize(1) shared(update_rows)
        for (int job = 0; job < jobs; ++job) {
            const int first = static_cast<int>(tiles * std::sqrt(static_cast<double>(job) / jobs));
            const int last  = static_cast<int>(tiles * std::sqrt(static_cast<double>(job + 1) / jobs));
            for (int tile = first; tile < last; ++tile) { update_rows(tile); }
        }
    } else
#endif
    {
        for (int tile = 0; tile < tiles; ++tile) { update_rows(tile); }
    }

#endif
}

} // namespace mf::dense
