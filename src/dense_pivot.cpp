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

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
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
    int                         dimension,
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
        double& target = values[static_cast<std::size_t>(row)*dimension+column];
        target -= correction;
        if (row != column) { values[static_cast<std::size_t>(column)*dimension+row] = target; }
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
                        double* target = values+static_cast<std::size_t>(row+i)*dimension+column;
                        tile[i] = update_vector(target, u0, u1, lower[p+i]);
                        _mm256_storeu_pd(target, tile[i]);
                    }
                    const __m256d a = _mm256_unpacklo_pd(tile[0], tile[1]);
                    const __m256d b = _mm256_unpackhi_pd(tile[0], tile[1]);
                    const __m256d c = _mm256_unpacklo_pd(tile[2], tile[3]);
                    const __m256d d = _mm256_unpackhi_pd(tile[2], tile[3]);
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column)*dimension+row,   _mm256_permute2f128_pd(a,c,0x20));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+1)*dimension+row, _mm256_permute2f128_pd(b,d,0x20));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+2)*dimension+row, _mm256_permute2f128_pd(a,c,0x31));
                    _mm256_storeu_pd(values+static_cast<std::size_t>(column+3)*dimension+row, _mm256_permute2f128_pd(b,d,0x31));
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
                double* target = values+static_cast<std::size_t>(row)*dimension+column;
                const __m256d result = update_vector(target, u0, u1, lower[p]);
                _mm256_storeu_pd(target, result);
                alignas(32) double mirror[4];
                _mm256_store_pd(mirror, result);
                for (int i = 0; i < 4; ++i) {
                    if (column+i != row) { values[static_cast<std::size_t>(column+i)*dimension+row] = mirror[i]; }
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

/** Dispatch one immediate symmetric update without allocating kernel scratch. */
void symmetric_pivot_update(
    int                         dimension,
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
    if (rank == 1) {
        symmetric_pivot_update_impl<1>(
            dimension,
            row_count,
            rows,
            lower,
            column_count,
            columns,
            upper0,
            upper1,
            values
        );
    } else if (rank == 2) {
        symmetric_pivot_update_impl<2>(
            dimension,
            row_count,
            rows,
            lower,
            column_count,
            columns,
            upper0,
            upper1,
            values
        );
    } else {
        throw std::runtime_error("Symmetric pivot update requires rank one or two");
    }
}

} // namespace mf::dense
