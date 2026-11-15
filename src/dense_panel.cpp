/**
 * @file dense_panel.cpp
 * @brief Assembles and factorizes exact multifrontal pivot panels.
 *
 * A 4-by-4 transpose reads contribution rows contiguously and writes contiguous
 * CSC segments. Scalar POTRF handles the small diagonal block. AVX2 TRSM holds
 * eight rows and four pivot columns in registers, reusing each previous factor
 * vector across four updates. The trailing Schur operation is implemented by
 * the existing dense kernels in the neighboring module.
 */
#include "dense_panel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {

/**
 * Assemble all remaining pivot columns from one packed contribution traversal.
 *
 * The first pivot column was separated by extend-add. Thus packed origin maps
 * to local front index one. Full tiles transpose four packed rows into factor
 * columns; diagonal and boundary strips use the same scalar additions.
 */
void add_packed_panel(int size, int width, double* const* columns, const double* packed, int origin) {
    // The first column is complete, and width-one fronts have no panel to add
    if (width <= 1) { return; }
    auto scalar_rows = [&](int first, int last, int first_column) {
        for (int r = first; r < last; ++r) {
            const std::size_t row = static_cast<std::size_t>(origin+r-1);
            const std::size_t base = row*(row+1)/2+origin;
            for (int j = first_column; j < width && j <= r; ++j) {
                columns[j][r-j] += packed[base+j-1];
            }
        }
    };
    int r = 1;
#if defined(__AVX2__) && defined(__FMA__)
    // The first three rows intersect the small leading triangular block
    scalar_rows(1,std::min(size,4),1);
    r = 4;
    auto tile = [&](int row, int column, int stored_columns) {
        const std::size_t a = static_cast<std::size_t>(origin+row-1);
        const __m256d x0 = _mm256_loadu_pd(packed+a*(a+1)/2+origin+column-1);
        const __m256d x1 = _mm256_loadu_pd(packed+(a+1)*(a+2)/2+origin+column-1);
        const __m256d x2 = _mm256_loadu_pd(packed+(a+2)*(a+3)/2+origin+column-1);
        const __m256d x3 = _mm256_loadu_pd(packed+(a+3)*(a+4)/2+origin+column-1);
        const __m256d t0 = _mm256_unpacklo_pd(x0,x1);
        const __m256d t1 = _mm256_unpackhi_pd(x0,x1);
        const __m256d t2 = _mm256_unpacklo_pd(x2,x3);
        const __m256d t3 = _mm256_unpackhi_pd(x2,x3);
        const __m256d values[] = {
            _mm256_permute2f128_pd(t0,t2,0x20),
            _mm256_permute2f128_pd(t1,t3,0x20),
            _mm256_permute2f128_pd(t0,t2,0x31),
            _mm256_permute2f128_pd(t1,t3,0x31)
        };
        for (int j = 0; j < stored_columns; ++j) {
            double* target = columns[column+j]+row-column-j;
            _mm256_storeu_pd(target,_mm256_add_pd(_mm256_loadu_pd(target),values[j]));
        }
    };
    for (; r+4 <= size; r += 4) {
        int j = 1;
        if (width >= 4) {
            // Store columns 1..3; column four is read but handled by the next
            // block/diagonal strip. All four source entries exist for row>=4.
            tile(r,1,3);
            j = 4;
            for (; j+4 <= width && j+3 <= r; j += 4) { tile(r,j,4); }
        }
        scalar_rows(r,r+4,j);
    }
#endif
    // Portable path and incomplete output-row tiles retain scalar addressing
    scalar_rows(r,size,1);
}

/**
 * Compute L11=chol(F11) in the leading block of the scalar CSC columns.
 *
 * The small POTRF base case preserves ascending correction order, normalizes
 * only leading-block entries, and publishes reciprocal diagonals for TRSM.
 * No pivot replacement or diagonal shift is introduced on an SPD failure.
 */
int factor_pivot_block(int width, double* const* columns, double* inverse, double& minimum) {
    // Factor the small dense leading block without touching the suffix yet
    minimum = std::numeric_limits<double>::infinity();
    for (int j = 0; j < width; ++j) {
        double diagonal = columns[j][0];
        for (int p = 0; p < j; ++p) {
            const double value = columns[p][j-p];
            diagonal -= value*value;
        }
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) { return j; }
        minimum = std::min(minimum,diagonal);
        columns[j][0] = std::sqrt(diagonal);
        inverse[j] = 1.0/columns[j][0];
        for (int r = j+1; r < width; ++r) {
            double value = columns[j][r-j];
            for (int p = 0; p < j; ++p) { value -= columns[p][r-p]*columns[p][j-p]; }
            columns[j][r-j] = value*inverse[j];
        }
    }
    return -1;
}

/**
 * Compute L21=F21*inv(L11^T) after the leading block has been factorized.
 *
 * Four target columns and eight right-hand sides stay in AVX2 registers while
 * each previous factor vector is reused four times. Local triangular coupling
 * follows the same ascending column order as the scalar reference. A scalar
 * path covers portable builds and incomplete row/column tiles.
 */
void solve_panel(int size, int width, double* const* columns, const double* inverse) {
    auto scalar_rows = [&](int first_row, int last_row, int first_column, int last_column) {
        for (int j = first_column; j < last_column; ++j) {
            for (int r = first_row; r < last_row; ++r) {
                double value = columns[j][r-j];
                for (int p = 0; p < j; ++p) { value -= columns[p][r-p]*columns[p][j-p]; }
                columns[j][r-j] = value*inverse[j];
            }
        }
    };
    int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
    // Four output columns share each previous eight-row factor vector
    for (; j+4 <= width; j += 4) {
        const __m256d inv0 = _mm256_set1_pd(inverse[j]);
        const __m256d inv1 = _mm256_set1_pd(inverse[j+1]);
        const __m256d inv2 = _mm256_set1_pd(inverse[j+2]);
        const __m256d inv3 = _mm256_set1_pd(inverse[j+3]);
        int r = width;
        for (; r+8 <= size; r += 8) {
            __m256d x00 = _mm256_loadu_pd(columns[j]+r-j);
            __m256d x01 = _mm256_loadu_pd(columns[j]+r-j+4);
            __m256d x10 = _mm256_loadu_pd(columns[j+1]+r-j-1);
            __m256d x11 = _mm256_loadu_pd(columns[j+1]+r-j+3);
            __m256d x20 = _mm256_loadu_pd(columns[j+2]+r-j-2);
            __m256d x21 = _mm256_loadu_pd(columns[j+2]+r-j+2);
            __m256d x30 = _mm256_loadu_pd(columns[j+3]+r-j-3);
            __m256d x31 = _mm256_loadu_pd(columns[j+3]+r-j+1);
            for (int p = 0; p < j; ++p) {
                const __m256d b0 = _mm256_loadu_pd(columns[p]+r-p);
                const __m256d b1 = _mm256_loadu_pd(columns[p]+r-p+4);
                __m256d a = _mm256_set1_pd(columns[p][j-p]);
                x00 = _mm256_fnmadd_pd(b0,a,x00); x01 = _mm256_fnmadd_pd(b1,a,x01);
                a = _mm256_set1_pd(columns[p][j+1-p]);
                x10 = _mm256_fnmadd_pd(b0,a,x10); x11 = _mm256_fnmadd_pd(b1,a,x11);
                a = _mm256_set1_pd(columns[p][j+2-p]);
                x20 = _mm256_fnmadd_pd(b0,a,x20); x21 = _mm256_fnmadd_pd(b1,a,x21);
                a = _mm256_set1_pd(columns[p][j+3-p]);
                x30 = _mm256_fnmadd_pd(b0,a,x30); x31 = _mm256_fnmadd_pd(b1,a,x31);
            }
            // Load local coefficients only after the previous-panel loop. This
            // keeps its eight accumulators and source vectors within AVX2's
            // register budget instead of keeping six extra broadcasts alive.
            // Solve the local triangular 4-by-4 coupling in the same p order
            x00 = _mm256_mul_pd(x00,inv0); x01 = _mm256_mul_pd(x01,inv0);
            __m256d coupling = _mm256_set1_pd(columns[j][1]);
            x10 = _mm256_mul_pd(_mm256_fnmadd_pd(x00,coupling,x10),inv1);
            x11 = _mm256_mul_pd(_mm256_fnmadd_pd(x01,coupling,x11),inv1);
            coupling = _mm256_set1_pd(columns[j][2]);
            x20 = _mm256_fnmadd_pd(x00,coupling,x20); x21 = _mm256_fnmadd_pd(x01,coupling,x21);
            coupling = _mm256_set1_pd(columns[j+1][1]);
            x20 = _mm256_mul_pd(_mm256_fnmadd_pd(x10,coupling,x20),inv2);
            x21 = _mm256_mul_pd(_mm256_fnmadd_pd(x11,coupling,x21),inv2);
            coupling = _mm256_set1_pd(columns[j][3]);
            x30 = _mm256_fnmadd_pd(x00,coupling,x30); x31 = _mm256_fnmadd_pd(x01,coupling,x31);
            coupling = _mm256_set1_pd(columns[j+1][2]);
            x30 = _mm256_fnmadd_pd(x10,coupling,x30); x31 = _mm256_fnmadd_pd(x11,coupling,x31);
            coupling = _mm256_set1_pd(columns[j+2][1]);
            x30 = _mm256_mul_pd(_mm256_fnmadd_pd(x20,coupling,x30),inv3);
            x31 = _mm256_mul_pd(_mm256_fnmadd_pd(x21,coupling,x31),inv3);
            _mm256_storeu_pd(columns[j]+r-j,x00);     _mm256_storeu_pd(columns[j]+r-j+4,x01);
            _mm256_storeu_pd(columns[j+1]+r-j-1,x10); _mm256_storeu_pd(columns[j+1]+r-j+3,x11);
            _mm256_storeu_pd(columns[j+2]+r-j-2,x20); _mm256_storeu_pd(columns[j+2]+r-j+2,x21);
            _mm256_storeu_pd(columns[j+3]+r-j-3,x30); _mm256_storeu_pd(columns[j+3]+r-j+1,x31);
        }
        scalar_rows(r,size,j,j+4);
    }
#endif
    // Scalar remainder also implements the complete portable triangular solve
    scalar_rows(width,size,j,width);
}

} // namespace mf::dense
