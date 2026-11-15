/**
 * @file tiled.cpp
 * @brief Independently checks tiled Schur storage and full multifrontal factors.
 *
 * Packed lower triangles provide the dense-update reference, including untouched
 * leading indices and tile padding. Manufactured SPD systems check inheritance,
 * mixed-layout extend-add and complete factors at different panel widths.
 */
#include "dense.h"
#include "dense_tiled.h"
#include "factor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

/** Compare all stored entries, including padding, and complete SPD factors. */
int main() {
    try {
        std::vector<int> thread_counts{1};
#ifdef _OPENMP
        thread_counts.push_back(4);
#endif
        int comparisons = 0;
        // Exercise offsets that cut through both input panels and output tiles
        auto kernel = [&](int n, int k, int origin, bool parallel) {
            (void)parallel;
            const int dimension = origin + n;
            std::vector<double> packed(static_cast<std::size_t>(dimension) * (dimension + 1) / 2);
            std::vector<double> tiled(mf::dense::tiled_lower_size(dimension + 1), 0.0);
            for (int r = 0; r < dimension; ++r) {
                for (int c = 0; c <= r; ++c) {
                    const double value = std::sin(0.03 * r + 0.02 * c);
                    packed[static_cast<std::size_t>(r) * (r+1) / 2 + c] = value;
                    tiled[mf::dense::tiled_lower_offset(r+1,c+1)] = value;
                }
            }
            std::vector<std::vector<double>> columns(k, std::vector<double>(n));
            std::vector<const double*> pointers;
            for (int p = 0; p < k; ++p) {
                for (int r = 0; r < n; ++r) { columns[p][r] = std::sin(0.013*r + 0.17*p); }
                pointers.push_back(columns[p].data());
            }
            mf::dense::rank_k_subtract(n,k,pointers.data(),packed.data(),origin);
            std::vector<double> scratch;
#ifdef _OPENMP
            if (parallel) {
#pragma omp parallel num_threads(4)
                {
#pragma omp single
                    mf::dense::rank_k_subtract_tiled(n,k,pointers.data(),tiled.data(),origin,scratch);
                }
            } else
#endif
            { mf::dense::rank_k_subtract_tiled(n,k,pointers.data(),tiled.data(),origin,scratch); }
            std::vector<double> expected(tiled.size(), 0.0);
            for (int r = 0; r < dimension; ++r) {
                for (int c = 0; c <= r; ++c) {
                    expected[mf::dense::tiled_lower_offset(r+1,c+1)] = packed[static_cast<std::size_t>(r)*(r+1)/2+c];
                }
            }
            for (std::size_t i = 0; i < tiled.size(); ++i) {
                if (std::abs(tiled[i]-expected[i]) > 1e-11) { throw std::runtime_error("Tiled dense mismatch"); }
            }
        };
        for (int n : {1,7,31,32,33,65,129}) {
            for (int k : {1,7,32}) {
                for (int origin : {0,1,7,31,32,63}) { kernel(n,k,origin,false); }
            }
        }
        kernel(1301,32,31,true);

        // Dense inheritance and branching/mixed-size child contributions
        for (int kind = 0; kind < 4; ++kind) {
            const int separator = kind == 1 ? 511 : kind == 2 ? 520 : 543;
            mf::Matrix matrix;
            matrix.n = kind == 0 ? 605 : separator + 4;
            matrix.ptr.push_back(0);
            for (int c = 0; c < matrix.n; ++c) {
                for (int r = c; r < matrix.n; ++r) {
                    double value = r == c ? 4.0 : 0.0;
                    if (kind == 0) { value += 0.001 * std::sin(0.03*r) * std::sin(0.03*c); }
                    else if ((c == 0 && r == 1) || (c == 2 && r == 3)) { value = -0.02; }
                    else if (c < 4 && r >= 4 && (kind != 2 || c < 2 || r % 2 == 0)) { value = -0.001; }
                    else if (c >= 4 && r == c + 1) { value = -0.01; }
                    if (value != 0.0) { matrix.row.push_back(r); matrix.value.push_back(value); }
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto plan = mf::Symbolic::analyze(matrix);
            std::vector<double> exact(matrix.n);
            for (int i = 0; i < matrix.n; ++i) { exact[i] = 1.0 + std::sin(0.03*i); }
            const auto rhs = matrix.multiply(exact);
            for (int width : {7,32,64}) {
                for (int threads : thread_counts) {
                    mf::Factor reference;
                    reference.threads = threads;
                    reference.tiled_schur = false;
                    reference.update_block_size = width;
                    reference.factorize(matrix,plan);
                    mf::Factor candidate;
                    candidate.threads = threads;
                    candidate.update_block_size = width;
                    candidate.tiled_schur = true;
                    candidate.factorize(matrix,plan);
                    for (std::size_t i = 0; i < candidate.value.size(); ++i) {
                        if (std::abs(candidate.value[i]-reference.value[i]) > 1e-12) {
                            throw std::runtime_error("Tiled factor mismatch");
                        }
                    }
                    ++comparisons;
                    const auto solution = candidate.solve(plan,rhs);
                    for (int i = 0; i < matrix.n; ++i) {
                        if (std::abs(solution[i]-exact[i]) > 1e-11) { throw std::runtime_error("Tiled solve mismatch"); }
                    }
                }
            }
            // A late nonpositive pivot in a tiled chain must invalidate the factor.
            if (kind == 3) {
                matrix.value.back() = -1.0;
                mf::Factor rejected_factor;
                bool rejected = false;
                try { rejected_factor.factorize(matrix, plan); }
                catch (const std::runtime_error&) { rejected = true; }
                if (!rejected || !rejected_factor.value.empty()) {
                    throw std::runtime_error("Failed tiled factor was retained");
                }
            }
        }
        std::cout << "PASS: 127 dense tile/origin cases, " << comparisons << " complete factor comparisons\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
