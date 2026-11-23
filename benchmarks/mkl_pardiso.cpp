/**
 * @file mkl_pardiso.cpp
 * @brief Independent oneMKL PARDISO timing reference for the same symmetric
 *        MatrixMarket operators used by FEMultiFrontal's LDLT benchmark.
 *
 * PARDISO mtype=-2 uses symmetric indefinite LDLT with possible perturbed
 * pivots. In particular, it does not certify the same numerical nullity as
 * FEMF's rank-aware solver. Report perturbations, inertia and original-system
 * residual, not only the execution time.
 *
 * Link dynamically against mkl_rt via benchmarks/build_mkl_pardiso.sh.
 */
#include "matrix.h"
#include <mkl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

MKL_INT checked_index(std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<MKL_INT>::max())) {
        throw std::runtime_error("Matrix exceeds MKL_INT indexing range; use matching ILP64 build");
    }
    return static_cast<MKL_INT>(size);
}

// Matrix is lower-triangular CSC. For a symmetric A, the same data constitute
// the UPPER triangle of CSR: column c in lower CSC is row c in upper CSR.
// Insert explicit zero diagonals where missing (some PARDISO configurations
// require structurally present diagonal entries).
struct UpperCsr {
    std::vector<MKL_INT> ia, ja;
    std::vector<double> values;
};
UpperCsr upper_csr(const mf::Matrix& a) {
    UpperCsr out;
    out.ia.reserve(static_cast<std::size_t>(a.n)+1);
    out.ia.push_back(0);
    out.values.reserve(a.value.size()+static_cast<std::size_t>(a.n));
    out.ja.reserve(a.row.size()+static_cast<std::size_t>(a.n));
    for (int row=0;row<a.n;++row) {
        const auto begin=a.ptr[row], end=a.ptr[row+1];
        if (begin==end || a.row[begin]!=row) {
            out.ja.push_back(row);
            out.values.push_back(0.0);
        }
        for (std::size_t k=begin;k<end;++k) {
            if (a.row[k]<row) {
                throw std::runtime_error("Input was not canonical lower CSC");
            }
            out.ja.push_back(checked_index(static_cast<std::size_t>(a.row[k])));
            out.values.push_back(a.value[k]);
        }
        out.ia.push_back(checked_index(out.ja.size()));
    }
    return out;
}

struct PardisoSolver {
    void* pt[64]{};
    MKL_INT iparm[64]{};
    MKL_INT maxfct=1, mnum=1, mtype=-2, n=0, nrhs=1, msglvl=0;
    MKL_INT error=0;
    std::vector<MKL_INT> perm;
    UpperCsr csr;
    std::vector<double> rhs, x;
    bool initialized=false;

    PardisoSolver(const mf::Matrix& a, int threads, bool matching, int perturb_exponent)
        : n(checked_index(static_cast<std::size_t>(a.n))),
          perm(static_cast<std::size_t>(a.n),0),
          csr(upper_csr(a)),
          rhs(static_cast<std::size_t>(a.n),0.0),
          x(static_cast<std::size_t>(a.n),0.0) {
        mkl_set_dynamic(0);
        mkl_set_num_threads(threads);
        pardisoinit(pt,&mtype,iparm);
        initialized=true;
        iparm[0]=1; // Keep initialized parameters; honor overrides below.
        iparm[34]=1; // C/zero-based CSR indexing.
        iparm[9]=perturb_exponent;
        if (matching) {
            iparm[10]=1; // Scaling with matching for indefinite systems.
            iparm[12]=1; // Weighted matching in the analysis phase.
        }
    }
    ~PardisoSolver() {
        if (!initialized) { return; }
        MKL_INT phase=-1;
        MKL_INT cleanup_error=0;
        pardiso(pt,&maxfct,&mnum,&mtype,&phase,&n,
                csr.values.data(),csr.ia.data(),csr.ja.data(),perm.data(),
                &nrhs,iparm,&msglvl,rhs.data(),x.data(),&cleanup_error);
    }
    void run(MKL_INT phase) {
        error=0;
        pardiso(pt,&maxfct,&mnum,&mtype,&phase,&n,
                csr.values.data(),csr.ia.data(),csr.ja.data(),perm.data(),
                &nrhs,iparm,&msglvl,rhs.data(),x.data(),&error);
        if (error!=0) {
            throw std::runtime_error("PARDISO phase="+std::to_string(phase)+
                                     " failed with error="+std::to_string(error));
        }
    }
};

double relative_residual(const mf::Matrix& a,
                         const std::vector<double>& x,
                         const std::vector<double>& rhs) {
    const auto ax=a.multiply(x);
    long double numerator=0, denominator=0;
    for (std::size_t i=0;i<ax.size();++i) {
        const auto d=static_cast<long double>(ax[i])-rhs[i];
        numerator+=d*d;
        denominator+=static_cast<long double>(rhs[i])*rhs[i];
    }
    return std::sqrt(static_cast<double>(numerator/std::max(denominator,1e-300L)));
}
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    const auto m=v.size()/2;
    return v.size()%2 ? v[m] : 0.5*(v[m-1]+v[m]);
}

} // namespace

int main(int argc,char** argv) {
    try {
        if (argc<2) {
            std::cerr<<"Usage: mkl_pardiso_bench matrix.mtx [--threads N]"
                        " [--repeat N] [--warmups N] [--matching]"
                        " [--pivot-perturb-exponent N]\n";
            return 2;
        }
        int threads=1, repeat=6, warmups=2, perturb=8;
        bool matching=false;
        for (int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--threads" && i+1<argc) { threads=std::stoi(argv[++i]); }
            else if (arg=="--repeat" && i+1<argc) { repeat=std::stoi(argv[++i]); }
            else if (arg=="--warmups" && i+1<argc) { warmups=std::stoi(argv[++i]); }
            else if (arg=="--matching") { matching=true; }
            else if (arg=="--pivot-perturb-exponent" && i+1<argc) {
                perturb=std::stoi(argv[++i]);
            } else { throw std::runtime_error("Unknown/incomplete argument: "+arg); }
        }
        if (threads<1 || repeat<1 || warmups<0 || perturb<1 || perturb>15) {
            throw std::runtime_error("Invalid thread, repeat, warmup or pivot-perturb settings");
        }

        char version[256]{};
        mkl_get_version_string(version,static_cast<int>(sizeof(version)));
        std::cout<<std::setprecision(9)<<std::unitbuf;
        std::cout<<"mkl_version="<<version<<"\n";

        const auto load_begin=Clock::now();
        const auto matrix=mf::Matrix::read_market(argv[1]);
        PardisoSolver solver(matrix,threads,matching,perturb);
        const auto load_end=Clock::now();
        // Same deterministic manufactured exact solution as FEMF's
        // symmetric pivot driver. The right-hand side is consistent for
        // singular matrices even if the solver perturbs their zero pivots.
        std::vector<double> exact(matrix.n);
        for (int i=0;i<matrix.n;++i) {
            exact[i]=std::sin(0.17*i)+0.5*std::cos(0.031*i);
        }
        solver.rhs=matrix.multiply(exact);

        std::cout<<"solver=mkl_pardiso matrix_type=symmetric_indefinite"
                 <<" mtype=-2 matrix_n="<<matrix.n
                 <<" upper_csr_nnz="<<solver.csr.ja.size()
                 <<" threads="<<threads<<" repeat="<<repeat
                 <<" warmups="<<warmups<<" matching="<<int(matching)
                 <<" perturb_exponent="<<perturb
                 <<" load_convert_s="<<seconds(load_begin,load_end)<<"\n";

        const auto analysis_begin=Clock::now();
        solver.run(11); // Ordering + symbolic analysis exactly once.
        const auto analysis_end=Clock::now();
        std::cout<<"analysis_s="<<seconds(analysis_begin,analysis_end)
                 <<" iparm_ordering="<<solver.iparm[1]
                 <<" iparm_matching="<<solver.iparm[12]
                 <<" iparm_scaling="<<solver.iparm[10]
                 <<" iparm_pivoting="<<solver.iparm[20]
                 <<" iparm_refinement="<<solver.iparm[7]<<"\n";

        std::vector<double> factors,solves;
        factors.reserve(repeat);
        solves.reserve(repeat);
        bool all_accurate=true;
        for (int trial=-warmups;trial<repeat;++trial) {
            const auto before=Clock::now();
            solver.run(22); // Numerical factorization; analysis is reused.
            const auto factored=Clock::now();
            solver.run(33); // Triangular solves + PARDISO refinement.
            const auto solved=Clock::now();
            const double residual=relative_residual(matrix,solver.x,solver.rhs);
            const bool accurate=std::isfinite(residual) && residual<=1e-8;
            all_accurate=all_accurate && accurate;
            if (trial<0) { continue; }
            const double ft=seconds(before,factored);
            const double st=seconds(factored,solved);
            factors.push_back(ft);
            solves.push_back(st);
            std::cout<<"trial="<<trial+1<<" factor_s="<<ft<<" solve_s="<<st
                     <<" relative_residual="<<residual
                     <<" residual_ok="<<int(accurate)
                     <<" perturbed_pivots="<<solver.iparm[13]
                     <<" inertia_positive="<<solver.iparm[21]
                     <<" inertia_negative="<<solver.iparm[22]
                     <<"\n";
        }
        const double fmed=median(factors), smed=median(solves);
        std::vector<double> diffs;
        for (const double t:factors) { diffs.push_back(std::abs(t-fmed)); }
        std::cout<<"factor_median_s="<<fmed<<" factor_MAD_s="<<median(diffs)
                 <<" solve_median_s="<<smed
                 <<" residuals_ok="<<int(all_accurate)
                 <<" perturbed_pivots="<<solver.iparm[13]
                 <<"\n";
        if (!all_accurate) {
            std::cerr<<"Warning: original-system residual exceeds 1e-8; "
                        "timing is not an accuracy-matched FEMF comparison.\n";
            return 3;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"MKL PARDISO benchmark failed: "<<e.what()<<"\n";
        return 1;
    }
}
