/**
 * @file ldlt_indexed_schur.cpp
 * @brief LLT-style packed factor panels for signed multifrontal LDLT updates.
 *
 * Both 4x8 and experimental 6x8 AVX2 microtiles keep the original physical packed Schur output.
 * Input factors are copied just once into separate rank-major 4-row and
 * 8-column panels. The noncontiguous-output case gathers only C's eight
 * active columns, never the whole C matrix, and scatters back to the original
 * physical positions. This matters when delayed pivots leave holes in active.
 */
#include "ldlt_indexed_schur.h"
#include "ldlt_schur.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf::dense {

template<int MR>
static void rank_k_ldlt_indexed_packed_impl(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed) {
    static_assert(MR == 4 || MR == 6);
    if (rank <= 0 || active.empty()) { return; }
#if !defined(__AVX2__) || !defined(__FMA__)
    rank_k_ldlt_tiled(origin, dim, active, rank, left, right, packed);
#else
    // Packing is not profitable for very small updates; the tested original
    // Schur kernel remains the portable fallback and is used for short panels.
    const int count = static_cast<int>(active.size());
    if (count < 96 || rank < 8) {
        rank_k_ldlt_tiled(origin, dim, active, rank, left, right, packed);
        return;
    }

    constexpr int NR = 8;
    const int row_tiles = (count+MR-1)/MR;
    const int col_tiles = (count+NR-1)/NR;
    const std::size_t a_len = static_cast<std::size_t>(row_tiles)*rank*MR;
    const std::size_t b_len = static_cast<std::size_t>(col_tiles)*rank*NR;
    std::vector<double> scratch(a_len+b_len+8);
    const auto addr = reinterpret_cast<std::uintptr_t>(scratch.data());
    double* b_data = scratch.data()+(((64-(addr&63))&63)/sizeof(double));
    double* a_data = b_data+b_len;

    // Pack only L and R. The physical Schur triangle is never repacked.
    // Keeping B rank-major and contiguous for each column tile is essential
    // for high FMA throughput when C contains delayed-pivot gaps.
    for (int j=0; j<count; j+=NR) {
        const int n=std::min(NR,count-j);
        double* panel=b_data+static_cast<std::size_t>(j/NR)*rank*NR;
        for (int p=0; p<rank; ++p) {
            const double* src=right+static_cast<std::size_t>(p)*dim;
            double* dst=panel+static_cast<std::size_t>(p)*NR;
            for (int t=0; t<n; ++t) { dst[t]=src[active[j+t]]; }
        }
    }
    for (int i=0; i<count; i+=MR) {
        const int n=std::min(MR,count-i);
        double* panel=a_data+static_cast<std::size_t>(i/MR)*rank*MR;
        for (int p=0; p<rank; ++p) {
            const double* src=left+static_cast<std::size_t>(p)*dim;
            double* dst=panel+static_cast<std::size_t>(p)*MR;
            for (int t=0; t<n; ++t) { dst[t]=src[active[i+t]]; }
        }
    }

    std::atomic<bool> nonfinite{false};
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d maxval = _mm256_set1_pd(std::numeric_limits<double>::max());
    auto update_rows = [&](int tile) {
        const int first=tile*MR;
        const int nr=std::min(MR,count-first);
        double* target[MR]={};
        for (int i=0; i<nr; ++i) {
            const std::size_t physical=static_cast<std::size_t>(origin+active[first+i]);
            target[i]=packed+physical*(physical+1)/2+origin;
        }
        const double* a_panel=a_data+static_cast<std::size_t>(tile)*rank*MR;
        int j=0;

        // Complete strictly subdiagonal 4x8 rectangles are independent.
        // Packed B is contiguous even when active output columns are not.
        if (nr==MR) {
            for (;j+NR<=first;j+=NR) {
                const int base=active[j];
                const bool contiguous=active[j+NR-1]==base+NR-1;
                __m256d c0[MR],c1[MR];
                __m128i ids0{},ids1{};
                if (!contiguous) {
                    ids0=_mm_setr_epi32(active[j],active[j+1],active[j+2],active[j+3]);
                    ids1=_mm_setr_epi32(active[j+4],active[j+5],active[j+6],active[j+7]);
                }
                for (int i=0;i<MR;++i) {
                    if (contiguous) {
                        c0[i]=_mm256_loadu_pd(target[i]+base);
                        c1[i]=_mm256_loadu_pd(target[i]+base+4);
                    } else {
                        c0[i]=_mm256_i32gather_pd(target[i],ids0,8);
                        c1[i]=_mm256_i32gather_pd(target[i],ids1,8);
                    }
                }
                const double* b_panel=b_data+static_cast<std::size_t>(j/NR)*rank*NR;
                for (int p=0;p<rank;++p) {
                    const __m256d u0=_mm256_loadu_pd(b_panel+static_cast<std::size_t>(p)*NR);
                    const __m256d u1=_mm256_loadu_pd(b_panel+static_cast<std::size_t>(p)*NR+4);
                    for (int i=0;i<MR;++i) {
                        const __m256d scale=_mm256_set1_pd(a_panel[static_cast<std::size_t>(p)*MR+i]);
                        c0[i]=_mm256_fnmadd_pd(scale,u0,c0[i]);
                        c1[i]=_mm256_fnmadd_pd(scale,u1,c1[i]);
                    }
                }
                for (int i=0;i<MR;++i) {
                    const __m256d q0=_mm256_cmp_pd(_mm256_andnot_pd(sign,c0[i]),maxval,_CMP_LE_OQ);
                    const __m256d q1=_mm256_cmp_pd(_mm256_andnot_pd(sign,c1[i]),maxval,_CMP_LE_OQ);
                    if (_mm256_movemask_pd(q0)!=15 || _mm256_movemask_pd(q1)!=15) {
                        nonfinite.store(true,std::memory_order_relaxed);
                        return;
                    }
                    if (contiguous) {
                        _mm256_storeu_pd(target[i]+base,c0[i]);
                        _mm256_storeu_pd(target[i]+base+4,c1[i]);
                    } else {
                        alignas(32) double tmp[NR];
                        _mm256_store_pd(tmp,c0[i]);
                        _mm256_store_pd(tmp+4,c1[i]);
                        for (int t=0;t<NR;++t) { target[i][active[j+t]]=tmp[t]; }
                    }
                }
            }
        }

        // Handle the strictly triangular band and any incomplete panels.
        // This remains exact for arbitrary active sets, including one active
        // row, rank-1/2 panels and arbitrary inherited origins.
        for (int i=0;i<nr;++i) {
            const int row=active[first+i];
            for (int q=j;q<=first+i;++q) {
                const int col=active[q];
                double value=target[i][col];
                for (int p=0;p<rank;++p) {
                    const std::size_t offset=static_cast<std::size_t>(p)*dim;
                    value-=left[offset+row]*right[offset+col];
                }
                if (!std::isfinite(value)) {
                    nonfinite.store(true,std::memory_order_relaxed);
                    return;
                }
                target[i][col]=value;
            }
        }
    };

#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads()>1 &&
        static_cast<double>(count)*count*rank>=5e7 && row_tiles>=8) {
        const int jobs=std::min(row_tiles,4*omp_get_num_threads());
#pragma omp taskloop grainsize(1) shared(update_rows)
        for (int job=0;job<jobs;++job) {
            const int first=static_cast<int>(row_tiles*std::sqrt(static_cast<double>(job)/jobs));
            const int last=static_cast<int>(row_tiles*std::sqrt(static_cast<double>(job+1)/jobs));
            for (int tile=first;tile<last;++tile) { update_rows(tile); }
        }
    } else
#endif
    {
        for (int tile=0;tile<row_tiles;++tile) { update_rows(tile); }
    }
    if (nonfinite.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Nonfinite indexed LDLT Schur update");
    }
#endif
}

void rank_k_ldlt_indexed_packed(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed) {
    rank_k_ldlt_indexed_packed_impl<4>(
        origin, dim, active, rank, left, right, packed);
}

void rank_k_ldlt_indexed_packed6(
    int origin, int dim, const std::vector<int>& active, int rank,
    const double* left, const double* right, double* packed) {
    rank_k_ldlt_indexed_packed_impl<6>(
        origin, dim, active, rank, left, right, packed);
}

} // namespace mf::dense
