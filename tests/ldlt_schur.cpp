/**
 * @file ldlt_schur.cpp
 * @brief Regression tests for signed tiled LDLT Schur updates.
 *
 * Covers both signs of D, negative/positive corrections, 2x2 rank columns,
 * active-index holes, inherited packed origins, SIMD tails and OpenMP tasks.
 */
#include "ldlt_schur.h"
#include "ldlt_indexed_schur.h"
#include "ldlt_tiled_front.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int dim, int rank, int origin, int layout, bool parallel_run) {
#ifndef _OPENMP
    (void)parallel_run;
#endif
    const int allocated = origin + dim + 7;
    std::vector<int> active;
    for (int i = 0; i < dim; ++i) {
        if ((layout == 1 && i%13 == 2) ||
            (layout == 2 && i%3 == 1) ||
            (layout == 3 && i%37 == 3)) { continue; }
        active.push_back(i);
    }
    std::vector<double> left(static_cast<std::size_t>(rank)*dim);
    std::vector<double> right(static_cast<std::size_t>(rank)*dim);
    for (int p = 0; p < rank; ++p) {
        for (int i = 0; i < dim; ++i) {
            const auto index = static_cast<std::size_t>(p)*dim+i;
            left[index] = std::sin(0.13*i+0.17*p)*0.3;
            right[index] = std::cos(0.07*i-0.11*p)*(p%2 == 0 ? 0.4 : -0.6);
        }
    }
    std::vector<double> expected(static_cast<std::size_t>(allocated)*(allocated+1)/2,1.25);
    for (int row : active) {
        const std::size_t physical = static_cast<std::size_t>(origin+row);
        const auto offset = physical*(physical+1)/2+origin;
        for (int col : active) {
            if (col > row) { break; }
            double v = expected[offset+col];
            for (int p = 0; p < rank; ++p) {
                const auto base = static_cast<std::size_t>(p)*dim;
                v -= left[base+row]*right[base+col];
            }
            expected[offset+col] = v;
        }
    }
    // Exercise incumbent tiled arithmetic, indexed packed panels and the
    // experimental PERSISTENT tile-major lower-triangle output. All inactive,
    // guard and upper-triangle positions must stay untouched.
    for (int variant=0; variant<3; ++variant) {
        const bool tiled_front = variant == 2;
        std::vector<double> actual(
            tiled_front ? mf::dense::tiled_front_entries(allocated) : expected.size(),
            1.25);
        auto update = [&] {
            if (variant==0) {
                mf::dense::rank_k_ldlt_tiled(
                    origin,dim,active,rank,left.data(),right.data(),actual.data());
            } else if (variant==1) {
                mf::dense::rank_k_ldlt_indexed_packed(
                    origin,dim,active,rank,left.data(),right.data(),actual.data());
            } else {
                mf::dense::rank_k_ldlt_tiled_front(
                    origin,dim,active,rank,left.data(),right.data(),actual.data());
            }
        };
#ifdef _OPENMP
        if (parallel_run) {
#pragma omp parallel num_threads(4)
#pragma omp single
            { update(); }
        } else
#endif
        { update(); }
        if (tiled_front) {
            // Every meaningful physical lower-triangle entry is compared
            // against an independent scalar row-packed reference. Other tile
            // padding, including the upper halves of diagonal tiles, must
            // retain the initialization sentinel exactly.
            std::vector<bool> valid(actual.size(),false);
            for (int r=0; r<allocated; ++r) {
                for (int c=0; c<=r; ++c) {
                    const auto off=mf::dense::tiled_front_offset(r,c);
                    valid[off]=true;
                    const double want=expected[static_cast<std::size_t>(r)*(r+1)/2+c];
                    const double diff=std::abs(actual[off]-want);
                    if (!std::isfinite(actual[off]) ||
                        diff>2e-12*std::max(1.0,std::abs(want))) {
                        throw std::runtime_error(
                            "Tile-packed frontal Schur differs at ("+
                            std::to_string(r)+","+std::to_string(c)+")");
                    }
                }
            }
            for (std::size_t i=0; i<actual.size(); ++i) {
                if (!valid[i] && actual[i] != 1.25) {
                    throw std::runtime_error("Tile-packed Schur touched padding");
                }
            }
        } else {
            for (std::size_t i=0; i<expected.size(); ++i) {
                const double diff=std::abs(actual[i]-expected[i]);
                if (!std::isfinite(actual[i]) ||
                    diff>2e-12*std::max(1.0,std::abs(expected[i]))) {
                    throw std::runtime_error(
                        std::string(variant==0?"Tiled":"Indexed packed")+
                        " LDLT Schur update differs from scalar reference at entry "+
                        std::to_string(i));
                }
            }
        }
    }
}

int main() {
    try {
        int cases = 0;
        for (int dim : {0,1,2,3,4,7,8,9,15,16,31,32,33,65,127,257}) {
            for (int rank : {1,2,8,32}) {
                for (int origin : {0,19}) {
                    for (int layout = 0; layout < 4; ++layout) {
                        check(dim,rank,origin,layout,false);
                        ++cases;
                    }
                }
            }
        }
        // Cross the intra-front parallel-work threshold with a contiguous
        // active set and a high enough rank, then exercise irregular holes.
        check(1025,64,19,0,true);
        check(1025,64,19,1,true);
        std::cout << "PASS: " << cases+2 << " tiled, indexed and tile-major packed LDLT Schur cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
