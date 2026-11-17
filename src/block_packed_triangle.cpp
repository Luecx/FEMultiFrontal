#include "block_packed_triangle.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {

void add_block_packed_panel(int size, int width, double* const* columns,
                            const BlockPackedTriangle& contribution, int origin) {
    for (int j = 1; j < width; ++j) {
        for (int r = j; r < size; ++r) {
            columns[j][r-j] += contribution.at(origin+r-1, origin+j-1);
        }
    }
}

void rank_k_subtract_bpt(int n, int k, const double* const* columns,
                         BlockPackedTriangle& contribution, int origin) {
    if (n <= 0 || k <= 0) { return; }
    const int first_block = origin / BlockPackedTriangle::tile_size;
    const int last_block = (origin + n - 1) / BlockPackedTriangle::tile_size;
    constexpr int B = BlockPackedTriangle::tile_size;
    const int active_end = origin + n;

    // Iterate native tiles. The origin can be INSIDE a tile, so old leading
    // entries must never be overwritten, including old diagonal tiles.
    for (int bi = first_block; bi <= last_block; ++bi) {
        const int rbase = bi * B;
        const int rbegin = std::max(origin, rbase);
        const int rend = std::min(active_end, rbase + contribution.block_length(bi));
        for (int bj = first_block; bj <= bi; ++bj) {
            const int cbase = bj * B;
            const int cbegin = std::max(origin, cbase);
            const int cend = std::min(active_end, cbase + contribution.block_length(bj));
            const int stride = contribution.block_length(bj);
            double* tile = contribution.block_data(bi, bj);
            for (int r = rbegin; r < rend; ++r) {
                const int last = bi == bj ? std::min(cend, r+1) : cend;
                double* out = tile + static_cast<std::size_t>(r-rbase) * stride + (cbegin-cbase);
                const int row = r-origin;
                int c = cbegin;
#if defined(__AVX2__) && defined(__FMA__)
                // Accumulate full RHS vector in registers across the rank loop.
                for (; c + 8 <= last; c += 8, out += 8) {
                    __m256d a = _mm256_loadu_pd(out);
                    __m256d b = _mm256_loadu_pd(out+4);
                    for (int p = 0; p < k; ++p) {
                        const __m256d x = _mm256_set1_pd(columns[p][row]);
                        a = _mm256_fnmadd_pd(x, _mm256_loadu_pd(columns[p]+c-origin), a);
                        b = _mm256_fnmadd_pd(x, _mm256_loadu_pd(columns[p]+c-origin+4), b);
                    }
                    _mm256_storeu_pd(out, a);
                    _mm256_storeu_pd(out+4, b);
                }
                for (; c + 4 <= last; c += 4, out += 4) {
                    __m256d a = _mm256_loadu_pd(out);
                    for (int p = 0; p < k; ++p) {
                        a = _mm256_fnmadd_pd(_mm256_set1_pd(columns[p][row]),
                                              _mm256_loadu_pd(columns[p]+c-origin), a);
                    }
                    _mm256_storeu_pd(out, a);
                }
#endif
                for (; c < last; ++c, ++out) {
                    double v = *out;
                    for (int p = 0; p < k; ++p) { v -= columns[p][row] * columns[p][c-origin]; }
                    *out = v;
                }
            }
        }
    }
}

} // namespace mf::dense
