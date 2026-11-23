/**
 * @file ldlt_kernel_sweep.cpp
 * @brief Standalone 20-way microbenchmark for FEMF's actual packed LDLT update.
 *
 * Workload: C(active,active) -= L(active,:) * R(active,:)^T, where
 * R = L*D and D contains signed 1x1 and indefinite 2x2 blocks.
 * Only the lower triangle in the inherited packed storage is touched.
 * The 6x8 packed version mirrors the *layout* of LLT's packed6 kernel,
 * but necessarily keeps separate L and R panels.
 *
 * This benchmark intentionally does not link to FEMF or to BLAS. It can
 * validate and time the kernels independently before changing solver code.
 */
#include "ldlt_schur.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace {
struct Fixture {
    int n=0, rank=0, origin=0;
    std::vector<int> active;
    std::vector<double> left, right, initial;
    std::size_t front_size() const { return initial.size(); }
    double* packed_row(double* c,int r) const {
        const std::size_t physical=static_cast<std::size_t>(origin+r);
        return c+physical*(physical+1)/2+origin;
    }
};
struct Scratch { std::vector<double> buffer; };
using Fn=void(*)(const Fixture&,double*,Scratch&);
struct Variant { const char* name; Fn call; };

Fixture make_fixture(int n,int rank,int holes,int origin) {
    Fixture f; f.n=n;f.rank=rank;f.origin=origin;
    for(int i=0;i<n;++i) {
        if(holes && (i*37+11)%100<holes)continue;
        f.active.push_back(i);
    }
    if(f.active.empty())throw std::runtime_error("Empty active front");
    f.left.resize(static_cast<std::size_t>(n)*rank);
    f.right.resize(f.left.size());
    for(int k=0;k<rank;++k) for(int i=0;i<n;++i) {
        f.left[static_cast<std::size_t>(k)*n+i]=
            0.012*std::sin(0.13*i+0.19*k)+0.003*std::cos(0.07*i-0.11*k);
    }
    // Genuine signed (potentially indefinite) D, with cross terms in 2x2.
    for(int k=0;k<rank;k+=2) {
        const double d00=(k%4==0?1.4:-1.2);
        if(k+1<rank) {
            const double d11=(k%4==0?-0.9:0.8),d01=0.4;
            for(int i=0;i<n;++i) {
                const double a=f.left[static_cast<std::size_t>(k)*n+i];
                const double b=f.left[static_cast<std::size_t>(k+1)*n+i];
                f.right[static_cast<std::size_t>(k)*n+i]=d00*a+d01*b;
                f.right[static_cast<std::size_t>(k+1)*n+i]=d01*a+d11*b;
            }
        } else {
            for(int i=0;i<n;++i) {
                f.right[static_cast<std::size_t>(k)*n+i]=
                    d00*f.left[static_cast<std::size_t>(k)*n+i];
            }
        }
    }
    const std::size_t physical=static_cast<std::size_t>(origin+n+7);
    f.initial.resize(physical*(physical+1)/2);
    for(std::size_t i=0;i<f.initial.size();++i)
        f.initial[i]=0.15*std::sin(0.001*static_cast<double>(i))+1.0;
    return f;
}

inline double cell(const Fixture& f,double* c,int r,int col) {
    double value=f.packed_row(c,r)[col];
    for(int k=0;k<f.rank;++k) {
        const std::size_t base=static_cast<std::size_t>(k)*f.n;
        value-=f.left[base+r]*f.right[base+col];
    }
    return value;
}
void scalar_reference(const Fixture& f,double* c,Scratch&) {
    for(std::size_t i=0;i<f.active.size();++i) {
        const int r=f.active[i];
        double* target=f.packed_row(c,r);
        for(std::size_t j=0;j<=i;++j)
            target[f.active[j]]=cell(f,c,r,f.active[j]);
    }
}
// Exact incumbent kernel, rather than a reconstructed approximation.
void production_current(const Fixture& f,double* c,Scratch&) {
    mf::dense::rank_k_ldlt_tiled(f.origin,f.n,f.active,f.rank,
                                 f.left.data(),f.right.data(),c);
}

template<int W> void column_unroll(const Fixture& f,double* c,Scratch&) {
    for(std::size_t i=0;i<f.active.size();++i) {
        const int row=f.active[i];
        double* target=f.packed_row(c,row);
        std::size_t j=0;
        for(;j+W<=i+1;j+=W) {
            double sum[W];
            for(int t=0;t<W;++t)sum[t]=target[f.active[j+t]];
            for(int k=0;k<f.rank;++k) {
                const std::size_t off=static_cast<std::size_t>(k)*f.n;
                const double a=f.left[off+row];
                for(int t=0;t<W;++t)
                    sum[t]-=a*f.right[off+f.active[j+t]];
            }
            for(int t=0;t<W;++t)target[f.active[j+t]]=sum[t];
        }
        for(;j<=i;++j)target[f.active[j]]=cell(f,c,row,f.active[j]);
    }
}

// PACK=0: unpacked input operands
// PACK=1: rank-major 8/16-wide right panels, constructed once per call
// PACK=2: independent rank-major A and B panels, as in LLT packed6
template<int MR,int NR,int PACK>
void tiled(const Fixture& f,double* c,Scratch& scratch) {
    static_assert(MR>0 && NR%4==0);
    const int na=static_cast<int>(f.active.size());
    const int ra=f.rank;
    const int a_panels=(na+MR-1)/MR, b_panels=(na+NR-1)/NR;
    const double* a_data=nullptr;
    const double* b_data=nullptr;
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr(PACK>0) {
        const std::size_t b_len=static_cast<std::size_t>(b_panels)*ra*NR;
        const std::size_t a_len=PACK==2?static_cast<std::size_t>(a_panels)*ra*MR:0;
        scratch.buffer.resize(b_len+a_len+8);
        const auto address=reinterpret_cast<std::uintptr_t>(scratch.buffer.data());
        double* aligned=scratch.buffer.data()+(((64-(address&63))&63)/sizeof(double));
        double* b=aligned;
        double* a=b+b_len;
        for(int j=0;j<na;j+=NR) {
            const int last=std::min(NR,na-j);
            const std::size_t block=static_cast<std::size_t>(j/NR)*ra*NR;
            for(int k=0;k<ra;++k) {
                double* dest=b+block+static_cast<std::size_t>(k)*NR;
                const double* src=f.right.data()+static_cast<std::size_t>(k)*f.n;
                for(int t=0;t<last;++t)dest[t]=src[f.active[j+t]];
            }
        }
        if constexpr(PACK==2) {
            for(int i=0;i<na;i+=MR) {
                const int last=std::min(MR,na-i);
                const std::size_t block=static_cast<std::size_t>(i/MR)*ra*MR;
                for(int k=0;k<ra;++k) {
                    double* dest=a+block+static_cast<std::size_t>(k)*MR;
                    const double* src=f.left.data()+static_cast<std::size_t>(k)*f.n;
                    for(int t=0;t<last;++t)dest[t]=src[f.active[i+t]];
                }
            }
        }
        b_data=b;
        if constexpr(PACK==2)a_data=a;
    }
#else
    (void)scratch;(void)a_data;(void)b_data;
#endif

    // One output row tile at a time; physical row-packed C is never repacked.
    for(int ib=0;ib<na;ib+=MR) {
        const int nr=std::min(MR,na-ib);
        double* target[MR]={};
        for(int i=0;i<nr;++i)target[i]=f.packed_row(c,f.active[ib+i]);
        int jb=0;
        if(nr==MR) for(;jb+NR<=ib;jb+=NR) {
            const int base=f.active[jb];
            const bool contiguous=f.active[jb+NR-1]==base+NR-1;
#if defined(__AVX2__) && defined(__FMA__)
            if(contiguous) {
                __m256d acc[MR][NR/4];
                for(int i=0;i<MR;++i)
                    for(int v=0;v<NR/4;++v)
                        acc[i][v]=_mm256_loadu_pd(target[i]+base+4*v);
                const std::size_t aoff=static_cast<std::size_t>(ib/MR)*ra*MR;
                const std::size_t boff=static_cast<std::size_t>(jb/NR)*ra*NR;
                for(int k=0;k<ra;++k) {
                    __m256d b[NR/4];
                    const double* rhs=PACK>0
                        ? b_data+boff+static_cast<std::size_t>(k)*NR
                        : f.right.data()+static_cast<std::size_t>(k)*f.n+base;
                    for(int v=0;v<NR/4;++v)b[v]=_mm256_loadu_pd(rhs+4*v);
                    for(int i=0;i<MR;++i) {
                        const double value=PACK==2
                            ? a_data[aoff+static_cast<std::size_t>(k)*MR+i]
                            : f.left[static_cast<std::size_t>(k)*f.n+f.active[ib+i]];
                        const __m256d x=_mm256_set1_pd(value);
                        for(int v=0;v<NR/4;++v)
                            acc[i][v]=_mm256_fnmadd_pd(x,b[v],acc[i][v]);
                    }
                }
                for(int i=0;i<MR;++i)
                    for(int v=0;v<NR/4;++v)
                        _mm256_storeu_pd(target[i]+base+4*v,acc[i][v]);
                continue;
            }
#else
            (void)contiguous;
#endif
            // Exact scalar fallback when active gaps prevent contiguous C loads.
            for(int i=0;i<MR;++i)
                for(int j=0;j<NR;++j) {
                    const int col=f.active[jb+j];
                    target[i][col]=cell(f,c,f.active[ib+i],col);
                }
        }
        // Triangular diagonal block and incomplete noncontiguous tails.
        for(int i=0;i<nr;++i)
            for(int j=jb;j<=ib+i;++j) {
                const int col=f.active[j];
                target[i][col]=cell(f,c,f.active[ib+i],col);
            }
    }
}

constexpr std::array<Variant,20> kernels{{
    {"scalar_reference",scalar_reference},
    {"column_unroll_2",column_unroll<2>},
    {"column_unroll_4",column_unroll<4>},
    {"column_unroll_8",column_unroll<8>},
    {"column_unroll_16",column_unroll<16>},
    {"direct_4x8",tiled<4,8,0>},
    {"direct_6x8",tiled<6,8,0>},
    {"direct_8x8",tiled<8,8,0>},
    {"direct_4x16",tiled<4,16,0>},
    {"direct_6x16",tiled<6,16,0>},
    {"direct_8x16",tiled<8,16,0>},
    {"direct_12x8",tiled<12,8,0>},
    {"direct_4x32",tiled<4,32,0>},
    {"production_current",production_current},
    {"packedB_4x8",tiled<4,8,1>},
    {"packedB_6x8",tiled<6,8,1>},
    {"packedB_6x16",tiled<6,16,1>},
    {"packedAB_4x8",tiled<4,8,2>},
    {"packedAB_6x8",tiled<6,8,2>},
    {"packedAB_6x16",tiled<6,16,2>}
}};

std::vector<int> numbers(const std::string& value) {
    std::vector<int> result;
    std::size_t first=0;
    while(first<value.size()) {
        const auto last=value.find(',',first);
        const auto term=value.substr(first,last==std::string::npos?last:last-first);
        const int n=std::stoi(term);
        result.push_back(n);
        if(last==std::string::npos)break;
        first=last+1;
    }
    if(result.empty())throw std::runtime_error("Empty number list");
    return result;
}
double validate(const Fixture& f) {
    auto baseline=f.initial;
    Scratch work;
    scalar_reference(f,baseline.data(),work);
    double error=0;
    for(const auto& kernel:kernels) {
        auto result=f.initial;
        kernel.call(f,result.data(),work);
        for(std::size_t j=0;j<result.size();++j) {
            const double ref=baseline[j],test=result[j];
            if(!std::isfinite(test))
                throw std::runtime_error(std::string(kernel.name)+": nonfinite result");
            const double relative=std::abs(test-ref)/std::max(1.0,std::abs(ref));
            if(relative>1e-12)
                throw std::runtime_error(std::string(kernel.name)+
                    ": wrong packed entry "+std::to_string(j)+
                    ", error="+std::to_string(relative));
            error=std::max(error,relative);
        }
    }
    return error;
}
struct Options {
    std::vector<int> sizes{256,768},ranks{32},holes{0,12},origins{19};
    int rounds=3,target_ms=30,cold_mib=96;
    std::string modes="hot,cold";
    bool verify_only=false;
};
Options parse(int argc,char** argv) {
    Options x;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        auto arg=[&]()->std::string {
            if(++i>=argc)throw std::runtime_error("Missing argument for "+key);
            return argv[i];
        };
        if(key=="--sizes")x.sizes=numbers(arg());
        else if(key=="--ranks")x.ranks=numbers(arg());
        else if(key=="--holes")x.holes=numbers(arg());
        else if(key=="--origins")x.origins=numbers(arg());
        else if(key=="--rounds")x.rounds=std::stoi(arg());
        else if(key=="--target-ms")x.target_ms=std::stoi(arg());
        else if(key=="--cold-mib")x.cold_mib=std::stoi(arg());
        else if(key=="--modes")x.modes=arg();
        else if(key=="--verify-only")x.verify_only=true;
        else if(key=="--help") {
            std::cout<<"LDLT 20-kernel sweep\n"
                <<"  --sizes 256,768 --ranks 32 --holes 0,12 --origins 19\n"
                <<"  --modes hot,cold --rounds 3 --target-ms 30 --cold-mib 96\n"
                <<"  --verify-only (all kernels must match scalar reference)\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: "+key);
    }
    if(x.rounds<1||x.target_ms<1||x.cold_mib<1)
        throw std::runtime_error("Invalid timing options");
    for(int s:x.sizes)if(s<1||s>8192)throw std::runtime_error("Size out of range");
    for(int k:x.ranks)if(k<1||k>256)throw std::runtime_error("Rank out of range");
    for(int k:x.holes)if(k<0||k>=100)throw std::runtime_error("Hole percentage out of range");
    for(int k:x.origins)if(k<0||k>1024)throw std::runtime_error("Origin out of range");
    if(x.modes!="hot"&&x.modes!="cold"&&x.modes!="hot,cold")
        throw std::runtime_error("Use --modes hot, cold or hot,cold");
    return x;
}
using Clock=std::chrono::steady_clock;
double elapsed_ms(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
volatile double checksum_sink=0.0;
double measure(const Fixture& f,const Variant& variant,const std::string& mode,
               int target_ms,int cold_mib) {
    Scratch work;
    auto warm=f.initial;
    const auto t0=Clock::now();
    variant.call(f,warm.data(),work);
    const auto t1=Clock::now();
    const double first=std::max(0.001,elapsed_ms(t0,t1));
    const std::size_t one=f.front_size();
    const bool cold=mode=="cold";
    // The ring is deliberately larger than a single CCD's LLC for large
    // enough fronts. Rotating buffers simulates independent Schur contributions.
    const std::size_t slots=cold
        ? std::max<std::size_t>(2,static_cast<std::size_t>(
            (static_cast<std::size_t>(cold_mib)<<20)/(one*sizeof(double))))
        : 1;
    const std::size_t iterations=std::max<std::size_t>(
        cold?slots:2,
        std::min<std::size_t>(100000,static_cast<std::size_t>(std::ceil(target_ms/first))));
    std::vector<double> pool;
    // Cap the count to avoid unreasonable allocation for tiny test fixtures.
    const std::size_t count=std::min<std::size_t>(slots,4096);
    pool.resize(one*count);
    for(std::size_t i=0;i<count;++i)
        std::copy(f.initial.begin(),f.initial.end(),pool.data()+i*one);
    // Cold workloads must rotate BOTH the contribution C and factor panels:
    // otherwise L/R stay hot and only C tests streaming memory behavior.
    // Fixture copies share no allocation with each other; initial is omitted.
    std::vector<Fixture> cold_factors;
    if(cold) {
        cold_factors.reserve(count);
        for(std::size_t i=0;i<count;++i) {
            Fixture candidate;
            candidate.n=f.n;candidate.rank=f.rank;candidate.origin=f.origin;
            candidate.active=f.active;
            candidate.left=f.left;
            candidate.right=f.right;
            cold_factors.push_back(std::move(candidate));
        }
    }
    const auto begin=Clock::now();
    for(std::size_t i=0;i<iterations;++i) {
        const std::size_t slot=i%count;
        const Fixture& factor=cold?cold_factors[slot]:f;
        variant.call(factor,pool.data()+slot*one,work);
    }
    const auto end=Clock::now();
    checksum_sink=pool[(iterations%count)*one+(one/2)];
    return elapsed_ms(begin,end)/iterations;
}
} // anonymous namespace

int main(int argc,char** argv) {
    try {
        const Options opts=parse(argc,argv);
        std::size_t tested=0;
        std::cout<<std::setprecision(9);
        if(!opts.verify_only)
            std::cout<<"size,rank,holes_pct,origin,mode,variant,median_ms,relative_to_scalar\n";
        for(int n:opts.sizes)for(int rank:opts.ranks)
            for(int hole:opts.holes)for(int origin:opts.origins) {
                const auto fixture=make_fixture(n,rank,hole,origin);
                const double error=validate(fixture);
                ++tested;
                if(opts.verify_only)continue;
                for(const std::string& mode:std::array<std::string,2>{"hot","cold"}) {
                    if(opts.modes=="hot"&&mode=="cold")continue;
                    if(opts.modes=="cold"&&mode=="hot")continue;
                    std::array<std::vector<double>,20> samples;
                    // Reverse evaluation order every other round to reduce
                    // turbo/temperature bias, as in paired A/B benchmarks.
                    for(int rep=0;rep<opts.rounds;++rep) {
                        for(int v=0;v<20;++v) {
                            const int index=rep%2==0?v:19-v;
                            samples[index].push_back(measure(fixture,kernels[index],mode,
                                opts.target_ms,opts.cold_mib));
                        }
                    }
                    std::array<double,20> medians{};
                    for(int v=0;v<20;++v) {
                        auto& times=samples[v];
                        std::sort(times.begin(),times.end());
                        medians[v]=times[times.size()/2];
                    }
                    for(int v=0;v<20;++v)
                        std::cout<<n<<','<<rank<<','<<hole<<','<<origin<<','<<mode<<','
                            <<kernels[v].name<<','<<medians[v]<<','
                            <<medians[0]/medians[v]<<'\n';
                    std::cerr<<"PASS n="<<n<<" rank="<<rank<<" holes="<<hole
                        <<" origin="<<origin<<" mode="<<mode
                        <<" max_error="<<error<<'\n';
                }
            }
        if(opts.verify_only)
            std::cout<<"PASS "<<tested<<" fixtures x "<<kernels.size()
                <<" LDLT signed packed update kernels\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL: "<<e.what()<<'\n';
        return 1;
    }
}
