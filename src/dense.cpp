/**
 * @file dense.cpp
 * @brief Implements contiguous row-packed symmetric Schur-complement updates.
 *
 * A single scalar Cholesky pivot contributes minus the outer product of its
 * normalized off-diagonal column. Each packed row is contiguous and can be
 * vectorized by the compiler without an external numerical library.
 */
#include "dense.h"

#include <cstddef>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {

/**
 * Subtract the symmetric outer product from a packed lower triangle.
 *
 * Each logical row has i+1 entries, keeping the inner loop contiguous. origin
 * skips already eliminated rows/columns in an inherited contribution buffer.
 * The x and packed arrays must not overlap; n counts remaining update variables.
 */
void rank_one_subtract(int n, const double* x, double* packed, int origin) {
    // Apply the scalar pivot's Schur update to each lower-triangular row
    std::size_t offset = static_cast<std::size_t>(origin) * (origin + 1) / 2 + origin;
    for (int i = 0; i < n; ++i) {
        const double scale = x[i];
        double*      target = packed + offset;
        for (int j = 0; j <= i; ++j) {
            target[j] -= scale * x[j];
        }
        offset += static_cast<std::size_t>(origin + i) + 1;
    }
}

/**
 * Apply batched Schur updates while keeping a 4-by-8 output tile in registers.
 *
 * Columns are suffixes of previously computed scalar factor columns. AVX2/FMA
 * handles strictly off-diagonal rectangles, while the short diagonal remainder
 * uses scalar accumulation. Each output tile is read and written only once per
 * batch. The fallback uses the same arithmetic without architecture intrinsics.
 */
void rank_k_subtract(int n, int k, const double* const* columns, double* packed, int origin) {
    // Preserve the simple contiguous rank-one path for unbatched contributions
    if (k == 1) {
        rank_one_subtract(n, columns[0], packed, origin);
        return;
    }

    // Process four packed rows together to reuse each loaded vector across rows
    for (int i = 0; i < n; i += 4) {
        int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        if (i + 4 <= n) {
            double* target[4];
            for (int r = 0; r < 4; ++r) {
                const std::size_t row = static_cast<std::size_t>(origin + i + r);
                target[r] = packed + row * (row + 1) / 2 + origin;
            }
            for (; j + 8 <= i; j += 8) {
                // Accumulate eight entries in each of four rows without intermediate stores
                __m256d c00 = _mm256_loadu_pd(target[0] + j);
                __m256d c01 = _mm256_loadu_pd(target[0] + j + 4);
                __m256d c10 = _mm256_loadu_pd(target[1] + j);
                __m256d c11 = _mm256_loadu_pd(target[1] + j + 4);
                __m256d c20 = _mm256_loadu_pd(target[2] + j);
                __m256d c21 = _mm256_loadu_pd(target[2] + j + 4);
                __m256d c30 = _mm256_loadu_pd(target[3] + j);
                __m256d c31 = _mm256_loadu_pd(target[3] + j + 4);
                for (int p = 0; p < k; ++p) {
                    const double* x = columns[p];
                    const __m256d b0 = _mm256_loadu_pd(x + j);
                    const __m256d b1 = _mm256_loadu_pd(x + j + 4);
                    const __m256d a0 = _mm256_set1_pd(x[i]);
                    const __m256d a1 = _mm256_set1_pd(x[i + 1]);
                    const __m256d a2 = _mm256_set1_pd(x[i + 2]);
                    const __m256d a3 = _mm256_set1_pd(x[i + 3]);
                    c00 = _mm256_fnmadd_pd(a0, b0, c00);
                    c01 = _mm256_fnmadd_pd(a0, b1, c01);
                    c10 = _mm256_fnmadd_pd(a1, b0, c10);
                    c11 = _mm256_fnmadd_pd(a1, b1, c11);
                    c20 = _mm256_fnmadd_pd(a2, b0, c20);
                    c21 = _mm256_fnmadd_pd(a2, b1, c21);
                    c30 = _mm256_fnmadd_pd(a3, b0, c30);
                    c31 = _mm256_fnmadd_pd(a3, b1, c31);
                }
                _mm256_storeu_pd(target[0] + j, c00);
                _mm256_storeu_pd(target[0] + j + 4, c01);
                _mm256_storeu_pd(target[1] + j, c10);
                _mm256_storeu_pd(target[1] + j + 4, c11);
                _mm256_storeu_pd(target[2] + j, c20);
                _mm256_storeu_pd(target[2] + j + 4, c21);
                _mm256_storeu_pd(target[3] + j, c30);
                _mm256_storeu_pd(target[3] + j + 4, c31);
            }
        }
#endif

        // Finish the diagonal band, partial row group, or portable fallback
        for (int r = i; r < i + 4 && r < n; ++r) {
            const std::size_t row = static_cast<std::size_t>(origin + r);
            double* target = packed + row * (row + 1) / 2 + origin;
            for (int c = j; c <= r; ++c) {
                double v = target[c];
                for (int p = 0; p < k; ++p) {
                    v -= columns[p][r] * columns[p][c];
                }
                target[c] = v;
            }
        }
    }
}

/**
 * Compute a sparse-column dot product against indirectly addressed x entries.
 *
 * Four AVX accumulators break the scalar reduction dependency while gather
 * loads the ordered sparse row indices. The final scalar tail covers short
 * columns. Reassociation changes rounding but does not alter the solve formula.
 */
double indexed_dot(int n, const int* indices, const double* values, const double* x) {
    // Accumulate sixteen sparse products per iteration with independent FMA chains
    int i = 0;
    double sum = 0.0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    __m256d s2 = _mm256_setzero_pd();
    __m256d s3 = _mm256_setzero_pd();
    for (; i + 16 <= n; i += 16) {
        const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices + i));
        const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices + i + 4));
        const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices + i + 8));
        const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(indices + i + 12));
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(values + i), _mm256_i32gather_pd(x, r0, 8), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(values + i + 4), _mm256_i32gather_pd(x, r1, 8), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(values + i + 8), _mm256_i32gather_pd(x, r2, 8), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(values + i + 12), _mm256_i32gather_pd(x, r3, 8), s3);
    }

    // Reduce the independent lanes once after all vector products
    const __m256d combined = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
    const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(combined), _mm256_extractf128_pd(combined, 1));
    sum = _mm_cvtsd_f64(_mm_hadd_pd(pair, pair));
#endif
    for (; i < n; ++i) {
        sum += values[i] * x[indices[i]];
    }
    return sum;
}

} // namespace mf::dense
