/**
 * @file pivot_update.cpp
 * @brief Checks immediate symmetric pivot updates independently of factorization.
 *
 * Scalar lower-triangle corrections supply the reference for signed scalar D
 * and indefinite 2-by-2 D blocks. Tests cover SIMD boundaries, active-index
 * holes, zero multipliers, symmetric mirrors and untouched inactive entries.
 */
#include "dense_pivot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

/** Compare all stored entries, not only the updated lower triangle. */
int main() {
    try {
        int cases = 0;
        for (int n : {0,1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,127}) {
            for (int rank : {1,2}) {
                for (int layout = 0; layout < 4; ++layout) {
                    for (int zeros : {0,1}) {
                        std::vector<int> columns, rows;
                        std::vector<std::array<double, 2>> lower;
                        std::vector<double> upper0(n,0.0), upper1(n,0.0);
                        for (int i = 0; i < n; ++i) {
                            if ((layout == 1 && i%7 == 3) || (layout == 2 && i%2 != 0) ||
                                (layout == 3 && i == n/2)) { continue; }
                            columns.push_back(i);
                            if (zeros != 0 && i%6 == 0) { continue; }
                            const double l0 = 0.2*std::sin(0.13*i+0.7);
                            const double l1 = rank == 2 ? 0.3*std::cos(0.17*i) : 0.0;
                            rows.push_back(i);
                            lower.push_back({l0,l1});
                            if (rank == 1) { upper0[i] = (layout%2 == 0 ? 2.0 : -2.0)*l0; }
                            else {
                                // D = [[0,2],[2,-0.5]] has both inertia signs
                                upper0[i] = 2.0*l1;
                                upper1[i] = 2.0*l0-0.5*l1;
                            }
                        }
                        std::vector<double> expected(static_cast<std::size_t>(n)*n);
                        for (int r = 0; r < n; ++r) {
                            for (int c = 0; c < n; ++c) { expected[r*n+c] = std::cos(0.09*(r+c)); }
                        }
                        auto actual = expected;

                        // Follow the original pivot loop, including its mirror
                        for (std::size_t p = 0; p < rows.size(); ++p) {
                            const int r = rows[p];
                            for (int c : columns) {
                                if (c > r) { continue; }
                                expected[r*n+c] -= lower[p][0]*upper0[c] + lower[p][1]*upper1[c];
                                if (r != c) { expected[c*n+r] = expected[r*n+c]; }
                            }
                        }
                        mf::dense::symmetric_pivot_update(
                            n, rank, static_cast<int>(rows.size()), rows.data(), lower.data(),
                            static_cast<int>(columns.size()), columns.data(), upper0.data(), upper1.data(), actual.data()
                        );
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i]) >
                                5e-14*std::max(1.0,std::abs(expected[i]))) {
                                throw std::runtime_error("Symmetric pivot update or inactive-entry mismatch");
                            }
                        }
                        ++cases;
                    }
                }
            }
        }

        // Compare task updates with the serial kernel on a strided inherited
        // front. The padded columns and inactive rows must stay untouched.
        {
            constexpr int live_width = 1025;
            constexpr int stride = live_width + 19;
            std::vector<int> active;
            std::vector<std::array<double,2>> lower;
            std::vector<double> upper0(live_width), upper1(live_width);
            for (int i = 0; i < live_width; ++i) {
                if (i % 17 == 5) { continue; }
                active.push_back(i);
                const double a = std::sin(0.07*i);
                const double b = std::cos(0.13*i);
                lower.push_back({a,b});
                upper0[i] = b;
                upper1[i] = a;
            }
            std::vector<double> serial(static_cast<std::size_t>(live_width)*stride, 2.0);
            auto parallel = serial;
            const int count = static_cast<int>(active.size());
            auto update = [&](double* values) {
                mf::dense::symmetric_pivot_update(
                    stride, 2, count, active.data(), lower.data(), count, active.data(),
                    upper0.data(), upper1.data(), values
                );
            };
            update(serial.data());
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
#pragma omp single
#endif
            { update(parallel.data()); }
            for (std::size_t i = 0; i < serial.size(); ++i) {
                if (!(std::abs(serial[i]-parallel[i]) <= 1e-14)) {
                    throw std::runtime_error("Parallel strided pivot update differs from serial");
                }
            }
            for (int r = 0; r < live_width; ++r) {
                for (int c = live_width; c < stride; ++c) {
                    if (parallel[static_cast<std::size_t>(r)*stride+c] != 2.0) {
                        throw std::runtime_error("Pivot update wrote into inactive storage");
                    }
                }
            }
        }

        // Compare the full panel with a sequence of immediate scalar updates.
        // A nonzero origin and an incomplete final tile cover the inherited
        // storage layout. The test checks the complete buffer, including guards.
        {
            constexpr int origin = 13, end = 333, stride = 349, pivots = 16;
            std::vector<double> lower(pivots*stride), upper(pivots*stride);
            std::vector<double> expected(stride*stride, 3.25);
            auto actual = expected;
            std::vector<int> active;
            for (int i = origin; i < end; ++i) { active.push_back(i); }
            std::vector<std::array<double,2>> multiplier(active.size());
            for (int k = 0; k < pivots; ++k) {
                for (int i = origin; i < end; ++i) {
                    const double value = std::sin(0.15*i+0.27*k)+1.25;
                    lower[k*stride+i] = value;
                    upper[k*stride+i] = (k%2 == 0 ? 0.5 : -0.5)*value;
                    multiplier[i-origin] = {value,0.0};
                }
                mf::dense::symmetric_pivot_update(
                    stride, 1, end-origin, active.data(), multiplier.data(), end-origin, active.data(),
                    upper.data()+k*stride, nullptr, expected.data()
                );
            }
#ifdef _OPENMP
#pragma omp parallel num_threads(4)
#pragma omp single
#endif
            {
                mf::dense::flush_scalar_pivot_updates(
                    stride, origin, end, pivots, lower.data(), upper.data(), actual.data()
                );
            }
            if (actual != expected) { throw std::runtime_error("Schur panel changed the update order or bounds"); }
        }

        // Check both scalar boundaries and the strict lower SIMD tile interior
        for (int rank : {1,2}) {
            for (int first_row : {0,4}) {
                for (double large : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1e308}) {
                    constexpr int n = 12;
                    std::vector<int> columns(n), rows(4);
                    std::vector<std::array<double, 2>> lower(4, {1e308,1e308});
                    std::vector<double> upper(n,large), values(n*n,0.0);
                    for (int i = 0; i < n; ++i) { columns[i] = i; }
                    for (int i = 0; i < 4; ++i) { rows[i] = first_row+i; }
                    bool rejected = false;
                    try {
                        mf::dense::symmetric_pivot_update(
                            n, rank, 4, rows.data(), lower.data(), n, columns.data(),
                            upper.data(), rank == 1 ? nullptr : upper.data(), values.data()
                        );
                    } catch (const std::runtime_error&) { rejected = true; }
                    if (!rejected) { throw std::runtime_error("Nonfinite pivot correction accepted"); }
                }
            }
        }
        bool rejected = false;
        try { mf::dense::symmetric_pivot_update(0,0,0,nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("Unsupported pivot rank accepted"); }
        std::cout << "PASS: " << cases << " symmetric pivot updates, mirrors, holes, tails and nonfinite rejection\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
