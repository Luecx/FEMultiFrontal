/**
 * @file ldlt_staged_gemm.cpp
 * @brief BLAS-3-style Schur update without modifying frontal storage format.
 *
 * Pack the two signed LDLT operands ONCE per flush. For each 64x64 output
 * tile, gather the physical lower-triangular entries to a dense scratch tile,
 * perform C -= A*B using either CBLAS dgemm (optional) or an AVX2/FMA 4x8
 * register kernel, then scatter only the meaningful lower entries back.
 *
 * This is intentionally a controlled A/B experiment: BLAS-3 work on dense C
 * incurs explicit staging traffic, but avoids the persistent layout and
 * contribution-compaction costs of the rejected tile-major-front experiment.
 */
#include "ldlt_staged_gemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(FEMF_USE_CBLAS)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#else
#error "WITH_CBLAS=1 requires cblas.h or mkl_cblas.h (set BLAS_CPPFLAGS)"
#endif
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mf::dense {
namespace {

constexpr int TILE = 64;

void internal_gemm(int m,int n,int k,const double* a,
                   const double* b,int ldb,double* c,int ldc) {
    // A is row-major [m,k], B is row-major [k,ldb], and C is
    // row-major [m,ldc]. A/B are already packed in active-index order.
    int i=0;
#if defined(__AVX2__) && defined(__FMA__)
    for (;i+4<=m;i+=4) {
        int j=0;
        for (;j+8<=n;j+=8) {
            __m256d c00=_mm256_loadu_pd(c+static_cast<std::size_t>(i)*ldc+j);
            __m256d c01=_mm256_loadu_pd(c+static_cast<std::size_t>(i)*ldc+j+4);
            __m256d c10=_mm256_loadu_pd(c+static_cast<std::size_t>(i+1)*ldc+j);
            __m256d c11=_mm256_loadu_pd(c+static_cast<std::size_t>(i+1)*ldc+j+4);
            __m256d c20=_mm256_loadu_pd(c+static_cast<std::size_t>(i+2)*ldc+j);
            __m256d c21=_mm256_loadu_pd(c+static_cast<std::size_t>(i+2)*ldc+j+4);
            __m256d c30=_mm256_loadu_pd(c+static_cast<std::size_t>(i+3)*ldc+j);
            __m256d c31=_mm256_loadu_pd(c+static_cast<std::size_t>(i+3)*ldc+j+4);
            for (int p=0;p<k;++p) {
                const double* const bptr=b+static_cast<std::size_t>(p)*ldb+j;
                const __m256d v0=_mm256_loadu_pd(bptr);
                const __m256d v1=_mm256_loadu_pd(bptr+4);
                const __m256d a0=_mm256_set1_pd(a[static_cast<std::size_t>(i)*k+p]);
                const __m256d a1=_mm256_set1_pd(a[static_cast<std::size_t>(i+1)*k+p]);
                const __m256d a2=_mm256_set1_pd(a[static_cast<std::size_t>(i+2)*k+p]);
                const __m256d a3=_mm256_set1_pd(a[static_cast<std::size_t>(i+3)*k+p]);
                c00=_mm256_fnmadd_pd(a0,v0,c00);
                c01=_mm256_fnmadd_pd(a0,v1,c01);
                c10=_mm256_fnmadd_pd(a1,v0,c10);
                c11=_mm256_fnmadd_pd(a1,v1,c11);
                c20=_mm256_fnmadd_pd(a2,v0,c20);
                c21=_mm256_fnmadd_pd(a2,v1,c21);
                c30=_mm256_fnmadd_pd(a3,v0,c30);
                c31=_mm256_fnmadd_pd(a3,v1,c31);
            }
            _mm256_storeu_pd(c+static_cast<std::size_t>(i)*ldc+j,c00);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i)*ldc+j+4,c01);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+1)*ldc+j,c10);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+1)*ldc+j+4,c11);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+2)*ldc+j,c20);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+2)*ldc+j+4,c21);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+3)*ldc+j,c30);
            _mm256_storeu_pd(c+static_cast<std::size_t>(i+3)*ldc+j+4,c31);
        }
        for (;j<n;++j) {
            for (int r=0;r<4;++r) {
                double& value=c[static_cast<std::size_t>(i+r)*ldc+j];
                for (int p=0;p<k;++p) {
                    value-=a[static_cast<std::size_t>(i+r)*k+p]*
                           b[static_cast<std::size_t>(p)*ldb+j];
                }
            }
        }
    }
#endif
    for (;i<m;++i) {
        for (int j=0;j<n;++j) {
            double& value=c[static_cast<std::size_t>(i)*ldc+j];
            for (int p=0;p<k;++p) {
                value-=a[static_cast<std::size_t>(i)*k+p]*
                       b[static_cast<std::size_t>(p)*ldb+j];
            }
        }
    }
}

void gemm(int m,int n,int k,const double* a,const double* b,int ldb,
          double* c,int ldc) {
#if defined(FEMF_USE_CBLAS)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                m,n,k,-1.0,a,k,b,ldb,1.0,c,ldc);
#else
    internal_gemm(m,n,k,a,b,ldb,c,ldc);
#endif
}

} // namespace

void rank_k_ldlt_staged_gemm(
    int origin,int dim,const std::vector<int>& active,int rank,
    const double* left,const double* right,double* packed,
    StagedGemmProfile* timings) {
    if (rank<=0 || active.empty()) { return; }
    using Clock=std::chrono::steady_clock;
    const auto started=timings ? Clock::now() : Clock::time_point{};
    const int count=static_cast<int>(active.size());
    // One packed A and B per complete panel flush, reused by all C tiles.
    // A[i,p] = left[p, active[i]], B[p,j] = right[p,active[j]].
    std::vector<double> a(static_cast<std::size_t>(count)*rank);
    std::vector<double> b(static_cast<std::size_t>(rank)*count);
    for (int p=0;p<rank;++p) {
        const std::size_t source=static_cast<std::size_t>(p)*dim;
        const std::size_t brow=static_cast<std::size_t>(p)*count;
        for (int i=0;i<count;++i) {
            b[brow+i]=right[source+active[i]];
            a[static_cast<std::size_t>(i)*rank+p]=left[source+active[i]];
        }
    }
    if (timings) {
        timings->pack_s+=std::chrono::duration<double>(Clock::now()-started).count();
    }

    // Retain the original frontal storage. Scratch C holds one 64x64 tile
    // in row-major form; diagonal tiles stage only their lower-triangle values.
    std::vector<double> c(static_cast<std::size_t>(TILE)*TILE);
    for (int ib=0;ib<count;ib+=TILE) {
        const int nr=std::min(TILE,count-ib);
        for (int jb=0;jb<=ib;jb+=TILE) {
            const int nc=std::min(TILE,count-jb);
            const bool diagonal=ib==jb;
            const int base=active[jb];
            const bool contiguous=(active[jb+nc-1]-base==nc-1);

            const auto load_begin=timings ? Clock::now() : Clock::time_point{};
            for (int i=0;i<nr;++i) {
                const int physical=origin+active[ib+i];
                const double* const source=packed+
                    static_cast<std::size_t>(physical)*(physical+1)/2+origin;
                double* const dst=c.data()+static_cast<std::size_t>(i)*TILE;
                const int lower=diagonal ? std::min(i+1,nc) : nc;
                if (contiguous) {
                    std::copy_n(source+base,lower,dst);
                } else {
                    for (int j=0;j<lower;++j) { dst[j]=source[active[jb+j]]; }
                }
                // Upper entries of a diagonal C tile are scratch-only. They
                // may be computed by GEMM but never written to packed C.
                if (lower<nc) { std::fill(dst+lower,dst+nc,0.0); }
            }
            if (timings) {
                timings->load_s+=std::chrono::duration<double>(
                    Clock::now()-load_begin).count();
            }

            const auto gemm_begin=timings ? Clock::now() : Clock::time_point{};
            gemm(nr,nc,rank,a.data()+static_cast<std::size_t>(ib)*rank,
                 b.data()+jb,count,c.data(),TILE);
            if (timings) {
                timings->gemm_s+=std::chrono::duration<double>(
                    Clock::now()-gemm_begin).count();
            }

            const auto store_begin=timings ? Clock::now() : Clock::time_point{};
            // Validate BEFORE changing any entries in the permanent front.
            // This includes all meaningful entries but not the scratch-only
            // upper triangle of a diagonal block.
            for (int i=0;i<nr;++i) {
                const int lower=diagonal ? std::min(i+1,nc) : nc;
                const double* const src=c.data()+static_cast<std::size_t>(i)*TILE;
                for (int j=0;j<lower;++j) {
                    if (!std::isfinite(src[j])) {
                        throw std::runtime_error("Nonfinite staged GEMM LDLT Schur update");
                    }
                }
            }
            for (int i=0;i<nr;++i) {
                const int physical=origin+active[ib+i];
                double* const target=packed+
                    static_cast<std::size_t>(physical)*(physical+1)/2+origin;
                const double* const src=c.data()+static_cast<std::size_t>(i)*TILE;
                const int lower=diagonal ? std::min(i+1,nc) : nc;
                if (contiguous) {
                    std::copy_n(src,lower,target+base);
                } else {
                    for (int j=0;j<lower;++j) { target[active[jb+j]]=src[j]; }
                }
            }
            if (timings) {
                timings->store_s+=std::chrono::duration<double>(
                    Clock::now()-store_begin).count();
                ++timings->tiles;
            }
        }
    }
}

} // namespace mf::dense
