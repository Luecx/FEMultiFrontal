/**
 * @file robustness.cpp
 * @brief Independently exercises SPD, threshold LDLT and LU numerical contracts.
 *
 * Dense manufactured operators provide original-system residual and normwise
 * backward-error checks independent of the solver's internal residual gate.
 * Rank-deficient solutions are judged by equations and the declared threshold,
 * not by comparison with a nonunique manufactured solution. This observer does
 * not modify pivot selection, delays, rank tolerances or arithmetic kernels.
 */
#include "factor.h"
#include "pivot_factor.h"
#include "ordering.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Run manufactured numerical cases through the existing public APIs at one and
 * four threads. Capture unsupported/failed cases as reportable failures, without
 * changing solver policy to make tests pass. Report factor and solve time with
 * accuracy, estimated rank, delayed pivots and inertia when applicable.
 */
int main() {
    using Clock = std::chrono::steady_clock;
    auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    std::cout << std::unitbuf << std::setprecision(12);
    int failures = 0;
    std::vector<int> thread_counts{1};
#ifdef _OPENMP
    thread_counts.push_back(4);
#endif
    auto check = [&](const std::string& name, int n, const std::vector<double>& dense,
                     bool symmetric, int expected_rank, bool incompatible = false) {
        // Build full CSC directly; no symmetrization can hide unsymmetric input
        mf::GeneralMatrix input;
        input.n = n;
        input.ptr.push_back(0);
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) {
                if (dense[r * n + c] != 0.0) {
                    input.row.push_back(r);
                    input.value.push_back(dense[r * n + c]);
                }
            }
            input.ptr.push_back(input.row.size());
        }
        const auto order = mf::nested_dissection(input.ordering_graph());
        const auto matrix = input.permute(order, order);
        std::vector<double> exact(n);
        for (int i = 0; i < n; ++i) { exact[i] = 1.0 + std::sin(0.7 * i); }
        auto rhs = matrix.multiply(exact);
        if (incompatible) { std::fill(rhs.begin(), rhs.end(), 1.0); }
        for (int threads : thread_counts) {
            const auto symbolic = mf::Symbolic::analyze(matrix.ordering_graph(), threads);
            mf::PivotFactor factor;
            factor.kind = symmetric ? mf::PivotFactor::Kind::Symmetric : mf::PivotFactor::Kind::General;
            factor.threads = threads;
            bool factored_ok = false;
            try {
                const auto begin = Clock::now();
                factor.factorize(matrix, symbolic);
                factored_ok = true;
                const auto factored = Clock::now();
                const auto x = factor.solve(rhs);
                const auto solved = Clock::now();

                // Compute residuals in long double from the original permuted A
                std::vector<long double> ax(n, 0.0L), row_norm(n, 0.0L);
                for (int c = 0; c < n; ++c) {
                    for (auto p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
                        ax[matrix.row[p]] += static_cast<long double>(matrix.value[p]) * x[c];
                        row_norm[matrix.row[p]] += std::abs(static_cast<long double>(matrix.value[p]));
                    }
                }
                long double r2 = 0.0L, b2 = 0.0L, rinf = 0.0L, binf = 0.0L, xinf = 0.0L;
                for (int i = 0; i < n; ++i) {
                    const long double r = ax[i] - rhs[i];
                    r2 += r * r; b2 += static_cast<long double>(rhs[i]) * rhs[i];
                    rinf = std::max(rinf, std::abs(r));
                    binf = std::max(binf, std::abs(static_cast<long double>(rhs[i])));
                    xinf = std::max(xinf, std::abs(static_cast<long double>(x[i])));
                }
                const long double anorm = *std::max_element(row_norm.begin(), row_norm.end());
                const double residual = std::sqrt(r2 / std::max(b2, 1e-300L));
                const double backward = rinf / std::max(anorm * xinf + binf, 1e-300L);
                bool pass = !incompatible && factor.rank == expected_rank &&
                                  std::isfinite(residual) && residual < 1e-8 && backward < 1e-8;
                if (name.rfind("forced_2x2",0) == 0 && factor.two_by_two_pivots == 0) { pass = false; }
                if (name.rfind("delayed_2x2",0) == 0 &&
                    (factor.two_by_two_pivots == 0 || factor.delayed_candidates == 0)) { pass = false; }
                if (name == "forced_2x2_dense" &&
                    (factor.positive_directions != n/2 || factor.negative_directions != n/2)) { pass = false; }
                if (name == "dense_null6" && factor.negative_directions != 0) { pass = false; }
                if (!pass) { ++failures; }
                std::cout << "case=" << name << " mode=" << (symmetric ? "ldlt" : "lu")
                          << " threads=" << threads << " status=" << (pass ? "PASS" : "FAIL")
                          << " n=" << n << " rank=" << factor.rank << " expected_rank=" << expected_rank
                          << " pivots2=" << factor.two_by_two_pivots << " delays=" << factor.delayed_candidates
                          << " max_front=" << factor.maximum_front
                          << " factor_s=" << seconds(begin, factored) << " solve_s=" << seconds(factored, solved)
                          << " residual=" << residual << " backward_error=" << backward << '\n';
            } catch (const std::exception& error) {
                const bool expected_rejection = incompatible && factored_ok;
                if (!expected_rejection) { ++failures; }
                std::cout << "case=" << name << " mode=" << (symmetric ? "ldlt" : "lu")
                          << " threads=" << threads << " status=" << (expected_rejection ? "PASS_REJECT" : "FAIL")
                          << " error=" << error.what() << '\n';
            }

            // Compare the fast Cholesky path on the same positive operators
            if (name.find("_spd") != std::string::npos || (symmetric && name.find("grid") == 0)) {
                mf::Matrix lower;
                lower.n = n;
                lower.ptr.push_back(0);
                for (int c = 0; c < n; ++c) {
                    for (auto p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
                        if (matrix.row[p] >= c) { lower.row.push_back(matrix.row[p]); lower.value.push_back(matrix.value[p]); }
                    }
                    lower.ptr.push_back(lower.row.size());
                }
                mf::Factor chol;
                chol.threads = threads;
                try {
                    const auto begin = Clock::now();
                    chol.factorize(lower, symbolic);
                    const auto factored = Clock::now();
                    const auto x = chol.solve(symbolic, rhs);
                    const auto solved = Clock::now();
                    const auto ax = matrix.multiply(x);
                    long double r2 = 0.0L, b2 = 0.0L;
                    for (int i = 0; i < n; ++i) {
                        const long double r = static_cast<long double>(ax[i]) - rhs[i];
                        r2 += r * r; b2 += static_cast<long double>(rhs[i]) * rhs[i];
                    }
                    const double residual = std::sqrt(r2 / b2);
                    const bool pass = std::isfinite(residual) && residual < 1e-8;
                    if (!pass) { ++failures; }
                    std::cout << "case=" << name << " mode=chol threads=" << threads << " status=" << (pass ? "PASS" : "FAIL")
                              << " n=" << n << " factor_s=" << seconds(begin, factored)
                              << " solve_s=" << seconds(factored, solved) << " residual=" << residual << '\n';
                } catch (const std::exception& error) {
                    ++failures;
                    std::cout << "case=" << name << " mode=chol threads=" << threads << " status=FAIL error=" << error.what() << '\n';
                }
            }
        }
    };

    // Finite well-conditioned and ill-conditioned symmetric positive operators
    check("well_spd", 3, {4,-1,0,-1,4,-1,0,-1,4}, true, 3);
    std::vector<double> hilbert(64);
    for (int r = 0; r < 8; ++r) { for (int c = 0; c < 8; ++c) { hilbert[r * 8 + c] = 1.0 / (r + c + 1.0); } }
    check("hilbert8_spd", 8, hilbert, true, 8);
    check("scaled_spd", 3, {1e-16,0,0,0,1,0,0,0,1e16}, true, 3);
    check("indefinite_diagonal", 3, {2,0,0,0,-3,0,0,0,1}, true, 3);
    check("forced_2x2", 2, {0,1,1,0}, true, 2);
    check("delayed_2x2", 4, {0,0,1,0,0,0,0,2,1,0,0,0.2,0,2,0.2,0}, true, 4);
    check("singular_laplacian", 3, {1,-1,0,-1,2,-1,0,-1,1}, true, 2);
    check("incompatible_laplacian", 3, {1,-1,0,-1,2,-1,0,-1,1}, true, 2, true);
    check("nearly_dependent", 2, {1,1,1,1+1e-14}, true, 1);
    check("unsymmetric_swaps", 3, {0,2,0,1,0,3,4,1,5}, false, 3);
    check("unsymmetric_scaled", 2, {1e-10,2e-10,-1,2}, false, 2);
    check("unsymmetric_singular", 3, {1,2,3,2,4,6,0,1,1}, false, 2);

    // Dense fronts exceed the SIMD cutoff and force an indefinite 2-by-2 block
    {
        constexpr int n = 96;
        std::vector<double> dense(n*n);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                dense[r*n+c] = r == c ? (r < 2 ? 0.0 : (r%2 ? 2.0 : -2.0)) : 0.001*std::cos(0.1*(r+c));
            }
        }
        dense[1] = dense[n] = 1.0;
        check("forced_2x2_dense",n,dense,true,n);

        // A strong future coupling postpones variable zero while several
        // intervening scalar pivots create holes in a still-large active front.
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                dense[r*n+c] = r == c ? ((r == 0 || r == 10) ? 0.0 : (r%2 ? 2.0 : -2.0)) : 0.001*std::cos(0.1*(r+c));
            }
        }
        dense[10] = dense[10*n] = 1.0;
        check("delayed_2x2_dense",n,dense,true,n);

        // A full-column-rank dyadic B gives A=B*B^T exactly rank n-6.
        // Its leading triangular block stays well conditioned during elimination.
        constexpr int rank = n-6;
        std::vector<double> basis(n*rank,0.0);
        for (int r = 0; r < n; ++r) {
            for (int k = 0; k < rank; ++k) {
                if (r < rank) { basis[r*rank+k] = r == k ? 1.0 : r > k ? 1.0/64 : 0.0; }
                else { basis[r*rank+k] = (1+(r+k)%3)/32.0; }
            }
        }
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                double value = 0.0;
                for (int k = 0; k < rank; ++k) { value += basis[r*rank+k]*basis[c*rank+k]; }
                dense[r*n+c] = value;
            }
        }
        check("dense_null6",n,dense,true,n-6);
        check("incompatible_dense_null6",n,dense,true,n-6,true);

        // Ill-conditioned full-rank SPD data must retain small finite pivots
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                double value = r == c ? 1e-8 : 0.0;
                for (int k = 0; k < 4; ++k) {
                    value += std::sin(0.13*r+0.7*k)*std::sin(0.13*c+0.7*k);
                }
                dense[r*n+c] = value;
            }
        }
        check("ill_conditioned96_spd",n,dense,true,n);
    }

    // Six axial bars of a free tetrahedron have six rigid-body null directions
    std::vector<double> tetra(144, 0.0);
    const double coordinates[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            double d[3];
            double norm = 0.0;
            for (int i = 0; i < 3; ++i) { d[i] = coordinates[b][i] - coordinates[a][i]; norm += d[i] * d[i]; }
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double value = d[i] * d[j] / norm;
                    tetra[(3*a+i)*12+3*a+j] += value;
                    tetra[(3*b+i)*12+3*b+j] += value;
                    tetra[(3*a+i)*12+3*b+j] -= value;
                    tetra[(3*b+i)*12+3*a+j] -= value;
                }
            }
        }
    }
    check("free_tetrahedron", 12, tetra, true, 6);

    // Moderate sparse grids expose the separate pivoted-path performance cost
    for (int width : {20, 40}) {
        const int n = width * width;
        for (bool symmetric : {true, false}) {
            std::vector<double> grid(static_cast<std::size_t>(n) * n, 0.0);
            for (int i = 0; i < n; ++i) {
                grid[i*n+i] = 5.0;
                for (int j : {i + 1, i + width}) {
                    if (j >= n || (j == i + 1 && j / width != i / width)) { continue; }
                    grid[i*n+j] = symmetric ? -1.0 : -0.8;
                    grid[j*n+i] = -1.0;
                }
            }
            check("grid" + std::to_string(width), n, grid, symmetric, n);
        }
    }
    std::cout << "failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
