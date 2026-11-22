/** @file ldlt_gemm.cpp
 * Packed-front Schur update via dense 24x64 tiles.
 *
 * The physical factor front stays packed (preserving extend-add and inherited
 * suffix semantics), but the arithmetic operates on dense, cache-local tiles.
 * Right-hand panels are packed in the K-major x 64 layout once per tile and
 * reused by all 24 output rows. This is a BLAS-3-like C -= L*R^T kernel:
 * it does NOT need SPD or nonnegative 1x1/2x2 pivots.
 */
#include "ldlt_gemm.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
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
void rank_k_ldlt_gemm(int origin, int dim, const std::vector<int>& active,
                      int rank, const double* left, const double* right,
                      double* packed) {
    if (rank <= 0 || active.empty()) { return; }
    constexpr int MR=24, NR=64;
    const int n=static_cast<int>(active.size());
    const int row_tiles=(n+MR-1)/MR;
    std::atomic<bool> nonfinite{false};
    auto work=[&](int row_tile) {
        const int rb=row_tile*MR, rn=std::min(MR,n-rb);
        alignas(64) double c[MR*NR];
        std::vector<double> rhs(static_cast<std::size_t>(rank)*NR);
        double* output[MR];
        for (int i=0;i<rn;++i) {
            const std::size_t row=static_cast<std::size_t>(origin+active[rb+i]);
            output[i]=packed+row*(row+1)/2+origin;
        }
        // Rectangular blocks entirely below the diagonal use the full dense
        // tile. The overlapping diagonal tile is masked only on the scatter.
        for (int cb=0;cb<rb+rn;cb+=NR) {
            const int cn=std::min(NR,rb+rn-cb);
            for (int k=0;k<rank;++k) {
                const double* from=right+static_cast<std::size_t>(k)*dim;
                double* to=rhs.data()+static_cast<std::size_t>(k)*NR;
                for (int j=0;j<cn;++j) { to[j]=from[active[cb+j]]; }
            }
            // Load a packed triangular tile into dense row-major scratch.
            for (int i=0;i<rn;++i) {
                double* dest=c+i*NR;
                for (int j=0;j<cn;++j) {
                    const int col=active[cb+j];
                    dest[j]=(cb+j<=rb+i)?output[i][col]:0.0;
                }
            }
            // All K columns are applied in acceptance order. Separate left
            // and right operands preserve signed/indefinite D exactly.
            for (int k=0;k<rank;++k) {
                const double* u=rhs.data()+static_cast<std::size_t>(k)*NR;
                const double* l=left+static_cast<std::size_t>(k)*dim;
                for (int i=0;i<rn;++i) {
                    const double a=l[active[rb+i]];
                    double* dest=c+i*NR;
                    int j=0;
#if defined(__AVX2__) && defined(__FMA__)
                    const __m256d av=_mm256_set1_pd(a);
                    for (;j+4<=cn;j+=4) {
                        const __m256d cv=_mm256_loadu_pd(dest+j);
                        const __m256d uv=_mm256_loadu_pd(u+j);
                        _mm256_storeu_pd(dest+j,_mm256_fnmadd_pd(av,uv,cv));
                    }
#endif
                    for (;j<cn;++j) { dest[j]-=a*u[j]; }
                }
            }
            // Write only actual active entries in the packed lower triangle.
            // Crucially, the part above the diagonal was never read from the
            // packed front and must never be written back.
            for (int i=0;i<rn;++i) {
                for (int j=0;j<cn && cb+j<=rb+i;++j) {
                    const double value=c[i*NR+j];
                    if (!std::isfinite(value)) {
                        nonfinite.store(true,std::memory_order_relaxed);
                        return;
                    }
                    output[i][active[cb+j]]=value;
                }
            }
        }
    };
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_num_threads()>1 &&
        static_cast<double>(n)*n*rank>=5e7 && row_tiles>=8) {
#pragma omp taskloop grainsize(1) shared(work)
        for (int t=0;t<row_tiles;++t) { work(t); }
    } else
#endif
    { for (int t=0;t<row_tiles;++t) { work(t); } }
    if (nonfinite.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Nonfinite dense-tile LDLT Schur update");
    }
}
} // namespace mf::dense
