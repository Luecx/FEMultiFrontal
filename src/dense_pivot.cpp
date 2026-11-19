/**
 * @file dense_pivot.cpp
 * @brief Implements immediate tiled LDLT updates with symmetric row-major writes.
 *
 * Four adjacent active rows and columns share pivot coefficients. Transposing
 * each updated 4-by-4 tile makes its upper-triangle mirror contiguous as well.
 * Irregular index sets use row-vector updates and scalar holes. Scalar fallback
 * supports portable builds, while AVX2/FMA handles full tiles without changing
 * pivot order or introducing a positive-definiteness assumption.
 */
#include "dense_pivot.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(_OPENMP) && defined(__AVX2__) && defined(__FMA__)
#include <omp.h>
#endif

namespace mf::dense {

/**
 * Apply one accepted pivot with compile-time rank and full symmetric storage.
 *
 * Coefficients already contain the action of the 1-by-1 or 2-by-2 D block.
 * Strictly lower 4-by-4 tiles are disjoint from their transposed destinations;
 * diagonal boundaries and holes retain scalar addressing. Every correction is
 * completed before returning, including mirroring and nonfinite validation.
 */
template<int Rank>
static void symmetric_pivot_update_impl(
    int                         stride,
    int                         row_count,
    const int*                  rows,
    const std::array<double, 2>* lower,
    int                         column_count,
    const int*                  columns,
    const double*               upper0,
    const double*               upper1,
    double*                     values
) {
    // Form the correction before subtracting it, as in the existing pivot loop.
    // Fusing the final subtraction can change values near pivot thresholds.
    // Scalar entries also cover holes and triangular boundaries
    auto update_entry = [&](int row, int column, const std::array<double, 2>& coefficient) {
        double correction;
#if defined(__FMA__)
        if constexpr (Rank == 1) { correction = std::fma(coefficient[0], upper0[column], 0.0); }
        else { correction = std::fma(coefficient[0], upper0[column], coefficient[1]*upper1[column]); }
#else
        if constexpr (Rank == 1) { correction = coefficient[0]*upper0[column] + 0.0; }
        else { correction = coefficient[0]*upper0[column] + coefficient[1]*upper1[column]; }
#endif
        double& target = values[static_cast<std::size_t>(row)*stride+column];
        target -= correction;
        if (row != column) { values[static_cast<std::size_t>(column)*stride+row] = target; }
        if (!std::isfinite(target)) { throw std::runtime_error("Nonfinite pivoted Schur update"); }
    };

#if defined(__AVX2__) && defined(__FMA__)
    const __m256d sign    = _mm256_set1_pd(-0.0);
    const __m256d maximum = _mm256_set1_pd(std::numeric_limits<double>::max());
    auto update_vector = [&](const double* source, __m256d u0, __m256d u1,
                             const std::array<double, 2>& coefficient) {
        __m256d correction;
        if constexpr (Rank == 1) {
            correction = _mm256_fmadd_pd(_mm256_set1_pd(coefficient[0]), u0, _mm256_setzero_pd());
        } else {
            correction = _mm256_fmadd_pd(
                _mm256_set1_pd(coefficient[0]), u0, _mm256_mul_pd(_mm256_set1_pd(coefficient[1]), u1)
            );
        }
        const __m256d result = _mm256_sub_pd(_mm256_loadu_pd(source), correction);
        const __m256d finite = _mm256_cmp_pd(_mm256_andnot_pd(sign, result), maximum, _CMP_LE_OQ);
        if (_mm256_movemask_pd(finite) != 15) { throw std::runtime_error("Nonfinite pivoted Schur update"); }
        return result;
    };
#endif
    int p = 0;
    while (p < row_count) {
#if defined(__AVX2__) && defined(__FMA__)
        // A full tile writes both triangles contiguously. Sorted distinct IDs
        // make endpoint differences sufficient to establish a consecutive run.
        if (p+4 <= row_count && rows[p+3]-rows[p] == 3) {
            const int row = rows[p];
            int index = 0;
            while (index < column_count && columns[index] < row) {
                const int column = columns[index];
                if (index+4 <= column_count && column+3 < row && columns[index+3]-column == 3) {
                    const __m256d u0 = _mm256_loadu_pd(upper0+column);
                    __m256d u1 = _mm256_setzero_pd();
                    if constexpr (Rank == 2) { u1 = _mm256_loadu_pd(upper1+column); }
                    __m256d tile[4];
                    for (int i = 0; i < 4; ++i) {
                        double* target = values+static_cast<std::size_t>(row+i)*stride+column;
                        tile[i] = update_vector(target, u0, u1, lower[p+i]);
                        _mm256_storeu_pd(target, tile[i]);
                    }
                    const __m256d a = _mm256_unpacklo_pd(tile[0], tile[1]);
                    const __m256d b = _mm256_unpackhi_pd(tile[0], tile[1]);
                    const __m256d c = _mm256_unpacklo_pd(tile[2], tile[3]);
                    const __m256d d = _mm256_unpackhi_pd(tile[2], tile[3]);
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column)*stride+row,   _mm256_permute2f128_pd(a,c,0x20));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+1)*stride+row, _mm256_permute2f128_pd(b,d,0x20));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+2)*stride+row, _mm256_permute2f128_pd(a,c,0x31));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+3)*stride+row, _mm256_permute2f128_pd(b,d,0x31));
                    index += 4;
                } else {
                    for (int i = 0; i < 4; ++i) { update_entry(row+i, column, lower[p+i]); }
                    ++index;
                }
            }
            // Remaining columns intersect this block's diagonal boundary
            for (int i = 0; i < 4; ++i) {
                for (int k = index; k < column_count && columns[k] <= row+i; ++k) {
                    update_entry(row+i, columns[k], lower[p+i]);
                }
            }
            p += 4;
            continue;
        }
#endif
        // Irregular rows still vectorize consecutive columns; holes stay scalar
        const int row = rows[p];
        int index = 0;
        while (index < column_count && columns[index] <= row) {
            const int column = columns[index];
#if defined(__AVX2__) && defined(__FMA__)
            if (index+4 <= column_count && column+3 <= row && columns[index+3]-column == 3) {
                const __m256d u0 = _mm256_loadu_pd(upper0+column);
                __m256d u1 = _mm256_setzero_pd();
                if constexpr (Rank == 2) { u1 = _mm256_loadu_pd(upper1+column); }
                double* target = values+static_cast<std::size_t>(row)*stride+column;
                const __m256d result = update_vector(target, u0, u1, lower[p]);
                _mm256_storeu_pd(target, result);
                alignas(32) double mirror[4];
                _mm256_store_pd(mirror, result);
                for (int i = 0; i < 4; ++i) {
                    if (column+i != row) { values[static_cast<std::size_t>(column+i)*stride+row] = mirror[i]; }
                }
                index += 4;
                continue;
            }
#endif
            update_entry(row, column, lower[p]);
            ++index;
        }
        ++p;
    }
}

/**
 * Complete the update in the existing factorization team for large fronts.
 *
 * Each task owns a disjoint range of lower-triangle rows and their mirrors.
 * The taskgroup joins before the front is read again or a nonfinite update
 * is reported to the caller.
 */
void symmetric_pivot_update(
    int                         stride,
    int                         rank,
    int                         row_count,
    const int*                  rows,
    const std::array<double, 2>* lower,
    int                         column_count,
    const int*                  columns,
    const double*               upper0,
    const double*               upper1,
    double*                     values
) {
    if (rank != 1 && rank != 2) {
        throw std::runtime_error("Symmetric pivot update requires rank one or two");
    }
    auto update_rows = [&](int first, int count) {
        if (rank == 1) {
            symmetric_pivot_update_impl<1>(
                stride, count, rows+first, lower+first, column_count, columns, upper0, upper1, values
            );
        } else {
            symmetric_pivot_update_impl<2>(
                stride, count, rows+first, lower+first, column_count, columns, upper0, upper1, values
            );
        }
    };

#if defined(_OPENMP) && defined(__AVX2__) && defined(__FMA__)
    // Estimate work from live entries. Square-root partitioning balances the
    // triangular work when active rows and columns mostly coincide.
    constexpr int entries_per_task = 128 * 1024;
    const auto work = static_cast<long long>(row_count) * column_count / 2;
    if (work >= 2*entries_per_task && omp_in_parallel()) {
        const int jobs = static_cast<int>(std::min<long long>(omp_get_num_threads(), work/entries_per_task));
        if (jobs > 1) {
            const int tiles = row_count/4;
            int invalid = 0;
#pragma omp taskloop grainsize(1) shared(update_rows) reduction(|:invalid)
            for (int job = 0; job < jobs; ++job) {
                const int first = 4*static_cast<int>(tiles*std::sqrt(static_cast<double>(job)/jobs));
                const int end = job+1 == jobs ? row_count :
                    4*static_cast<int>(tiles*std::sqrt(static_cast<double>(job+1)/jobs));
                try {
                    update_rows(first, end-first);
                } catch (...) {
                    invalid = 1;
                }
            }
            if (invalid) { throw std::runtime_error("Nonfinite pivoted Schur update"); }
            return;
        }
    }
#endif
    if (row_count > 0 && column_count > 0) { update_rows(0, row_count); }
}

} // namespace mf::dense
