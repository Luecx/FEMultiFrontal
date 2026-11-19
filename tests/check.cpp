/**
 * @file check.cpp
 * @brief Exercises frontal assembly, ordering, import and SPD failure behavior.
 *
 * Dense SPD factors are reconstructed independently, while a sparse grid checks
 * separator ordering and branch assembly. Disconnected and indefinite systems
 * cover elimination forests and the no-pivoting SPD contract.
 */
#include "factor.h"
#include "ordering.h"
#include "dense.h"
#include "dense_panel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Verify factor reconstruction and solutions across different elimination trees.
 *
 * Dense reference products are formed directly from sparse factor entries.
 * A manufactured nonconstant solution checks permutation and both substitutions
 * independently of the numerical assembly code.
 */
int main(int, char** argv) {
    try {
        // Cache-line alignment and optional page advice must preserve owned values
        for (std::size_t count : {std::size_t(1), std::size_t(17), std::size_t(300000)}) {
            mf::memory::Values values(count, 3.0);
            if (reinterpret_cast<std::uintptr_t>(values.data()) % 64 != 0) {
                throw std::runtime_error("Numerical storage is not cache-line aligned");
            }
            mf::memory::advise_huge_pages(values.data(), values.size() * sizeof(double));
            for (double value : values) {
                if (value != 3.0) { throw std::runtime_error("Page advice changed numerical storage"); }
            }
        }

        // Compare grouped substitution with independent scalar CSC traversal.
        // Cover every group width, vector tails, row gaps and untouched slots,
        // including the late RHS additions required by private subtree buffers.
        for (int width = 1; width <= 8; ++width) {
            for (int layout : {0, 1, 2}) {
                for (int size : {width, width+1, width+7, width+8, width+9, width+31, width+97, 257}) {
                    const int n = size * 3 + 11;
                    std::vector<int> rows(size);
                    for (int r = 0; r < size; ++r) {
                        rows[r] = layout == 0 ? 3*r+7 : layout == 1 ? r+7 : r+7+(r/13)*2;
                    }
                    std::vector<std::vector<double>> storage(width);
                    std::vector<const double*>       columns(width);
                    for (int j = 0; j < width; ++j) {
                        storage[j].resize(size-j);
                        for (int r = j; r < size; ++r) {
                            storage[j][r-j] = r == j ? 1.0+0.01*j : 0.013*std::sin(0.17*r+0.13*j);
                        }
                        columns[j] = storage[j].data();
                    }
                    for (bool late_rhs : {false, true}) {
                        std::vector<double> rhs(n), expected(n), actual(n);
                        for (int i = 0; i < n; ++i) {
                            rhs[i]      = std::cos(0.19*i);
                            expected[i] = 0.2*std::sin(0.11*i);
                            actual[i]   = expected[i];
                        }

                        // The reference follows each complete column in order
                        for (int j = 0; j < width; ++j) {
                            if (late_rhs) { expected[rows[j]] += rhs[rows[j]]; }
                            expected[rows[j]] /= columns[j][0];
                            const double pivot = expected[rows[j]];
                            for (int r = j+1; r < size; ++r) {
                                expected[rows[r]] -= columns[j][r-j] * pivot;
                            }
                        }
                        mf::dense::forward_panel(
                            size,
                            width,
                            rows.data(),
                            columns.data(),
                            actual.data(),
                            late_rhs ? rhs.data() : nullptr
                        );
                        for (int i = 0; i < n; ++i) {
                            if (!std::isfinite(actual[i]) || std::abs(expected[i]-actual[i]) > 1e-13) {
                                throw std::runtime_error("Grouped forward substitution differs from scalar columns");
                            }
                        }
                    }
                }
            }
        }

        // Reconstruct known dense Cholesky blocks across internal panel and
        // SIMD boundaries. The rectangular suffix must remain untouched.
        for (int width : {31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256}) {
            const int size = width + 9;
            std::vector<std::vector<double>> expected(width), original(width), values(width);
            std::vector<double*> columns(width);
            for (int c = 0; c < width; ++c) {
                expected[c].resize(width-c);
                for (int r = c; r < width; ++r) {
                    expected[c][r-c] = r == c ? 2.0+0.001*c : 0.002*std::sin(0.13*r+0.07*c);
                }
            }
            for (int c = 0; c < width; ++c) {
                original[c].resize(size-c, -123.0);
                for (int r = c; r < width; ++r) {
                    double value = 0.0;
                    for (int p = 0; p <= c; ++p) { value += expected[p][r-p]*expected[p][c-p]; }
                    original[c][r-c] = value;
                }
                values[c] = original[c];
                columns[c] = values[c].data();
            }
            for (int workers : {1, 4}) {
#ifndef _OPENMP
                if (workers != 1) { continue; }
#endif
                for (int c = 0; c < width; ++c) { std::copy(original[c].begin(), original[c].end(), values[c].begin()); }
                double inverse[256];
                double minimum;
                int bad = -2;
#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
#pragma omp single
                { bad = mf::dense::factor_pivot_block(width, columns.data(), inverse, minimum); }
#else
                bad = mf::dense::factor_pivot_block(width, columns.data(), inverse, minimum);
#endif
                if (bad != -1 || std::abs(minimum-4.0) > 1e-12) {
                    throw std::runtime_error("Blocked pivot factorization changed its SPD diagnostics");
                }
                for (int c = 0; c < width; ++c) {
                    if (std::abs(inverse[c]-1.0/expected[c][0]) > 1e-14) {
                        throw std::runtime_error("Blocked pivot inverse diagonal mismatch");
                    }
                    for (int r = c; r < size; ++r) {
                        const double reference = r < width ? expected[c][r-c] : original[c][r-c];
                        if (!std::isfinite(values[c][r-c]) || std::abs(values[c][r-c]-reference) > 1e-12) {
                            throw std::runtime_error("Blocked pivot reconstruction or suffix guard failed");
                        }
                    }
                }
            }
        }

        // SPD failures must retain the global pivot index across a panel boundary
        for (double diagonal : {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            constexpr int width = 65;
            std::vector<std::vector<double>> values(width);
            std::vector<double*> columns(width);
            for (int c = 0; c < width; ++c) {
                values[c].assign(width-c, 0.0);
                values[c][0] = c == 32 ? diagonal : 2.0;
                columns[c] = values[c].data();
            }
            double inverse[width];
            double minimum;
            if (mf::dense::factor_pivot_block(width, columns.data(), inverse, minimum) != 32) {
                throw std::runtime_error("Blocked pivot failure index mismatch");
            }
        }

        // The default grouping must stay at 128 when a Factor is reused with
        // different worker counts. Explicit widths retain identical arithmetic.
        {
            mf::Matrix matrix;
            matrix.n = 129;
            matrix.ptr.push_back(0);
            for (int c = 0; c < matrix.n; ++c) {
                for (int r = c; r < matrix.n; ++r) {
                    matrix.row.push_back(r);
                    matrix.value.push_back(r == c ? 4.0 : -0.001);
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto symbolic = mf::Symbolic::analyze(matrix);
            std::vector<double> exact(matrix.n);
            for (int i = 0; i < matrix.n; ++i) { exact[i] = std::sin(0.13*i) + 1.0; }
            const auto rhs = matrix.multiply(exact);
            mf::Factor fixed;
            constexpr int expected = 128;
            for (int workers : {1, 2, 4, 8, 16, 32, 64}) {
#ifdef _OPENMP
                if (workers > std::min(omp_get_num_procs(), omp_get_thread_limit())) { continue; }
#else
                if (workers != 1) { continue; }
#endif
                fixed.threads = workers;
                fixed.factorize(matrix, symbolic);
                if (fixed.update_block_size != expected) {
                    throw std::runtime_error("Default batch width changed with thread count");
                }
                if (fixed.update_block_size_used != expected) {
                    throw std::runtime_error("Default batch width changed under a runtime limit");
                }
                mf::Factor explicit_width;
                explicit_width.update_block_size = expected;
                explicit_width.factorize(matrix, symbolic);
                if (explicit_width.update_block_size_used != expected ||
                    fixed.assembled_fronts != explicit_width.assembled_fronts ||
                    fixed.merged_pivots != explicit_width.merged_pivots) {
                    throw std::runtime_error("Default and explicit front plans differ");
                }
                for (std::size_t i = 0; i < fixed.value.size(); ++i) {
                    if (std::abs(fixed.value[i]-explicit_width.value[i]) > 1e-13) {
                        throw std::runtime_error("Default and explicit factors differ");
                    }
                }
                const auto solution = fixed.solve(symbolic, rhs);
                for (int i = 0; i < matrix.n; ++i) {
                    if (std::abs(solution[i]-exact[i]) > 1e-12) {
                        throw std::runtime_error("Fixed batch solve changed the solution");
                    }
                }
            }
            for (int invalid : {-1, 0, 257}) {
                fixed.update_block_size = invalid;
                bool rejected = false;
                try { fixed.factorize(matrix, symbolic); }
                catch (const std::runtime_error&) { rejected = true; }
                if (!rejected || !fixed.value.empty()) {
                    throw std::runtime_error("Invalid batch size retained a usable factor");
                }
            }
        }

        // A nearly low-rank dense SPD operator stresses deferred column
        // corrections while requiring a small original-system residual despite
        // conditioning that can amplify forward error.
        {
            constexpr int n = 160;
            mf::Matrix matrix;
            matrix.n = n;
            matrix.ptr.push_back(0);
            for (int c = 0; c < n; ++c) {
                for (int r = c; r < n; ++r) {
                    double value = r == c ? 1e-6 : 0.0;
                    for (int p = 0; p < 6; ++p) {
                        const double a = std::sin(0.03 * r + 0.7 * p) + 0.2 * std::cos(0.17 * r - 0.11 * p);
                        const double b = std::sin(0.03 * c + 0.7 * p) + 0.2 * std::cos(0.17 * c - 0.11 * p);
                        value += a * b;
                    }
                    matrix.row.push_back(r);
                    matrix.value.push_back(value);
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto symbolic = mf::Symbolic::analyze(matrix);
            std::vector<double> exact(n);
            for (int i = 0; i < n; ++i) { exact[i] = 1.0 + std::sin(0.3 * i); }
            const auto rhs = matrix.multiply(exact);
            mf::Factor factor;
            factor.factorize(matrix, symbolic);
            const auto x = factor.solve(symbolic, rhs);
            const auto ax = matrix.multiply(x);
            long double residual2 = 0.0L, rhs2 = 0.0L;
            for (int i = 0; i < n; ++i) {
                const long double r = static_cast<long double>(ax[i]) - rhs[i];
                residual2 += r * r;
                rhs2 += static_cast<long double>(rhs[i]) * rhs[i];
                if (std::abs(x[i] - exact[i]) > 1e-5) { throw std::runtime_error("Ill-conditioned SPD forward error"); }
            }
            if (std::sqrt(residual2 / rhs2) > 1e-10L) { throw std::runtime_error("Ill-conditioned SPD residual"); }
        }

        // Exercise dense, 2D/3D grid, disconnected and node-blocked graphs with
        // every built ordering. The blocked volume grid has three unknowns per
        // node, except every seventh node which keeps two, so the ordering must
        // detect unequal indistinguishable groups from the pattern alone.
        std::vector<int> node_of_dof;
        for (int node = 0; node < 512; ++node) {
            for (int dof = 0; dof < (node % 7 == 0 ? 2 : 3); ++dof) { node_of_dof.push_back(node); }
        }
        auto adjacent_nodes = [](int a, int b) {
            const int dx = std::abs(a % 8 - b % 8);
            const int dy = std::abs(a / 8 % 8 - b / 8 % 8);
            const int dz = std::abs(a / 64 - b / 64);
            return dx + dy + dz <= 1;
        };
        const int sizes[] = {19, 225, 8, 504, static_cast<int>(node_of_dof.size())};
        for (int kind = 0; kind < 5; ++kind) {
            mf::Matrix matrix;
            matrix.n = sizes[kind];
            matrix.ptr.push_back(0);
            for (int c = 0; c < matrix.n; ++c) {
                for (int r = c; r < matrix.n; ++r) {
                    double v = 0.0;
                    if (kind == 4) {
                        // Coupled node blocks of a diagonally dominant volume operator
                        if (adjacent_nodes(node_of_dof[r], node_of_dof[c])) { v = r == c ? 40.0 : -0.5 - 0.01 * ((r + c) % 5); }
                    } else if (kind == 0) {
                        // A = R R^T + I with an explicitly constructed dense R
                        for (int k = 0; k < matrix.n; ++k) {
                            v += std::sin(0.7 * r + 0.13 * k) * std::sin(0.7 * c + 0.13 * k);
                        }
                        if (r == c) { v += 1.0; }
                    } else if (kind == 1) {
                        // Positive shifted five-point grid operator with a branching tree
                        if (r == c) { v = 5.0; }
                        if ((r == c + 1 && r / 15 == c / 15) || r == c + 15) { v = -1.0; }
                    } else if (kind == 2) {
                        if (r == c) { v = r + 1.0; }
                    } else {
                        // Shifted seven-point stencil on a 9-by-8-by-7 volume grid
                        if (r == c) { v = 7.0; }
                        if ((r == c + 1 && r / 9 == c / 9) ||
                            (r == c + 9 && r / 72 == c / 72) || r == c + 72) { v = -1.0; }
                    }
                    if (v != 0.0) { matrix.row.push_back(r); matrix.value.push_back(v); }
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            std::vector<int> methods{0, 1};
            for (int method : methods) {
                std::vector<int> order(matrix.n);
                std::iota(order.begin(), order.end(), 0);
                if (method == 1) {
                    order = mf::nested_dissection(matrix);
#ifdef _OPENMP
                    // Concurrent subdomain tasks must reproduce the serial permutation
                    if (mf::nested_dissection(matrix, 4) != order) {
                        throw std::runtime_error("Parallel nested dissection differs from the serial ordering");
                    }
#endif
                    // Detected groups must stay consecutive in their original order
                    if (kind == 4) {
                        std::vector<int> position(matrix.n);
                        for (int i = 0; i < matrix.n; ++i) { position[order[i]] = i; }
                        for (int d = 0; d + 1 < matrix.n; ++d) {
                            if (node_of_dof[d] == node_of_dof[d + 1] && position[d + 1] != position[d] + 1) {
                                throw std::runtime_error("Indistinguishable node unknowns were not kept consecutive");
                            }
                        }
                    }
                }
                const auto permuted = matrix.permute(order);
                const auto symbolic = mf::Symbolic::analyze(permuted);
#ifdef _OPENMP
                // Independent tree-reach counting must reproduce child-union
                // fronts exactly, including disconnected and variable-block cases.
                for (int threads : {2, 4}) {
                    const auto concurrent = mf::Symbolic::analyze(permuted, threads);
                    if (concurrent.ptr != symbolic.ptr || concurrent.row != symbolic.row ||
                        concurrent.parent != symbolic.parent || concurrent.reuse_child != symbolic.reuse_child ||
                        concurrent.first_child != symbolic.first_child || concurrent.next_child != symbolic.next_child ||
                        concurrent.max_front != symbolic.max_front || concurrent.factor_flops != symbolic.factor_flops ||
                        concurrent.postorder != symbolic.postorder || concurrent.subtree_begin != symbolic.subtree_begin ||
                        concurrent.subtree_size != symbolic.subtree_size || concurrent.subtree_work != symbolic.subtree_work ||
                        concurrent.roots != symbolic.roots) {
                        throw std::runtime_error("Parallel symbolic pattern differs from scalar-front unions");
                    }
                }
#endif
                mf::Factor factor;
                factor.factorize(permuted, symbolic);

                // Phase instrumentation must preserve the numerical factor and report finite totals
                const auto original_values = factor.value;
                factor.profile_enabled = true;
                factor.factorize(permuted, symbolic);
                for (std::size_t i = 0; i < factor.value.size(); ++i) {
                    if (std::abs(factor.value[i] - original_values[i]) > 1e-13 * std::max(1.0, std::abs(original_values[i]))) {
                        throw std::runtime_error("Instrumentation changed the numerical factor");
                    }
                }
                double measured = 0.0;
                for (double elapsed : factor.phase_seconds) {
                    if (!std::isfinite(elapsed) || elapsed < 0.0) {
                        throw std::runtime_error("Invalid phase timing");
                    }
                    measured += elapsed;
                }
                if (!(measured > 0.0)) { throw std::runtime_error("Missing phase timings"); }
                factor.profile_enabled = false;

                // Reconstruct L L^T without using frontal assembly
                const int n = matrix.n;
                std::vector<double> reconstructed(static_cast<std::size_t>(n) * n, 0.0);
                std::vector<double> expected(reconstructed.size(), 0.0);
                for (int c = 0; c < n; ++c) {
                    for (std::size_t i = symbolic.ptr[c]; i < symbolic.ptr[c + 1]; ++i) {
                        for (std::size_t j = symbolic.ptr[c]; j < symbolic.ptr[c + 1]; ++j) {
                            reconstructed[symbolic.row[i] * n + symbolic.row[j]] += factor.value[i] * factor.value[j];
                        }
                    }
                    for (std::size_t p = permuted.ptr[c]; p < permuted.ptr[c + 1]; ++p) {
                        expected[permuted.row[p] * n + c] = permuted.value[p];
                        expected[c * n + permuted.row[p]] = permuted.value[p];
                    }
                }
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    if (std::abs(expected[i] - reconstructed[i]) > 1e-11) {
                        throw std::runtime_error("L L^T reconstruction mismatch");
                    }
                }

                // Verify inverse permutation and both immediate and deferred Schur updates
                std::vector<double> exact(n);
                for (int i = 0; i < n; ++i) { exact[i] = std::sin(0.3 * i) + 1.0; }
                const auto rhs = matrix.multiply(exact);
                std::vector<double> permuted_rhs(n);
                for (int i = 0; i < n; ++i) { permuted_rhs[i] = rhs[order[i]]; }
                for (int block_size : {1, 8, 32, 64, 128, 256}) {
                    factor.update_block_size = block_size;
                    factor.factorize(permuted, symbolic);
                    for (double elapsed : factor.phase_seconds) {
                        if (elapsed != 0.0) { throw std::runtime_error("Disabled timing retained stale diagnostics"); }
                    }
                    const auto x = factor.solve(symbolic, permuted_rhs);
                    for (int i = 0; i < n; ++i) {
                        if (std::abs(x[i] - exact[order[i]]) > 1e-11) {
                            throw std::runtime_error("Manufactured solution mismatch");
                        }
                    }
                }
            }
        }

        // Exercise the adaptive packed path through a full factorization with a front above 512.
        // I + u*u^T is SPD by construction and includes non-multiple panel dimensions.
        {
            mf::Matrix matrix;
            matrix.n = 605;
            matrix.ptr.push_back(0);
            for (int c = 0; c < matrix.n; ++c) {
                for (int r = c; r < matrix.n; ++r) {
                    matrix.row.push_back(r);
                    matrix.value.push_back((r == c ? 1.0 : 0.0) + 0.0004 * std::sin(0.13 * c) * std::sin(0.13 * r));
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto symbolic = mf::Symbolic::analyze(matrix);
            std::vector<double> exact(matrix.n);
            for (int i = 0; i < matrix.n; ++i) { exact[i] = std::cos(0.17 * i); }
            const auto rhs = matrix.multiply(exact);
            mf::Factor factor;
            for (auto kernel : {mf::Factor::SchurKernel::Baseline, mf::Factor::SchurKernel::Packed}) {
                factor.schur_kernel = kernel;
                factor.factorize(matrix, symbolic);
                const auto x = factor.solve(symbolic, rhs);
                for (int i = 0; i < matrix.n; ++i) {
                    if (std::abs(x[i] - exact[i]) > 1e-11) { throw std::runtime_error("Adaptive packed solve mismatch"); }
                }
            }
        }

        // Compare AVX tiles, diagonal tails and nonzero packed origins against scalar arithmetic
        for (int n : {1, 3, 8, 17, 41, 64, 65, 127, 193}) {
            for (int origin : {0, 7}) {
                for (int k : {1, 3, 17, 32}) {
                    const int full = origin + n;
                    std::vector<double> actual(static_cast<std::size_t>(full) * (full + 1) / 2, 2.0);
                    auto expected = actual;
                    std::vector<std::vector<double>> columns(k, std::vector<double>(n));
                    std::vector<const double*> pointers;
                    for (int p = 0; p < k; ++p) {
                        for (int i = 0; i < n; ++i) { columns[p][i] = std::sin(0.1 * i + p); }
                        pointers.push_back(columns[p].data());
                    }
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j <= i; ++j) {
                            const std::size_t index = static_cast<std::size_t>(origin + i) * (origin + i + 1) / 2 + origin + j;
                            for (int p = 0; p < k; ++p) {
                                expected[index] -= columns[p][i] * columns[p][j];
                            }
                        }
                    }
                    std::vector<double> scratch;
                    for (int variant = 0; variant < 4; ++variant) {
                        std::fill(actual.begin(), actual.end(), 2.0);
                        if (variant == 0) { mf::dense::rank_k_subtract(n, k, pointers.data(), actual.data(), origin); }
                        if (variant == 1) { mf::dense::rank_k_subtract_wide(n, k, pointers.data(), actual.data(), origin); }
                        if (variant == 2) { mf::dense::rank_k_subtract_packed(n, k, pointers.data(), actual.data(), origin, scratch); }
                        if (variant == 3) { mf::dense::rank_k_subtract_packed6(n, k, pointers.data(), actual.data(), origin, scratch); }
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            if (std::abs(actual[i] - expected[i]) > 1e-12) {
                                throw std::runtime_error("Batched dense kernel mismatch");
                            }
                        }
                    }
                }
            }
        }

        // A dense change of reduced coordinates must not require physical DOF
        // groups. N spans the constraints u[8+i] = u[i], with a dense invertible
        // mixing of its eight independent coordinates; form A = N^T K N.
        {
            constexpr int physical = 12;
            constexpr int reduced  = 8;
            std::vector<double> basis(physical * reduced);
            for (int r = 0; r < reduced; ++r) {
                for (int c = 0; c < reduced; ++c) {
                    basis[r * reduced + c] = (r == c ? 1.0 : 0.0) + 0.02 * std::cos(0.2 * r + 0.3 * c);
                }
            }
            for (int r = reduced; r < physical; ++r) {
                for (int c = 0; c < reduced; ++c) { basis[r * reduced + c] = basis[(r - reduced) * reduced + c]; }
            }
            mf::Matrix transformed;
            transformed.n = reduced;
            transformed.ptr.push_back(0);
            for (int c = 0; c < reduced; ++c) {
                for (int r = c; r < reduced; ++r) {
                    double value = 0.0;
                    for (int i = 0; i < physical; ++i) {
                        value += 4.0 * basis[i * reduced + r] * basis[i * reduced + c];
                        if (i + 1 < physical) {
                            value -= basis[i * reduced + r] * basis[(i + 1) * reduced + c];
                            value -= basis[(i + 1) * reduced + r] * basis[i * reduced + c];
                        }
                    }
                    transformed.row.push_back(r);
                    transformed.value.push_back(value);
                }
                transformed.ptr.push_back(transformed.row.size());
            }
            const auto serial = mf::Symbolic::analyze(transformed);
            std::vector<int> counts{1};
#ifdef _OPENMP
            counts = {1, 4};
#endif
            std::vector<double> exact(reduced, 1.0);
            const auto rhs = transformed.multiply(exact);
            for (int threads : counts) {
                const auto plan = mf::Symbolic::analyze(transformed, threads);
                if (plan.ptr != serial.ptr || plan.row != serial.row || plan.parent != serial.parent ||
                    plan.reuse_child != serial.reuse_child || plan.postorder != serial.postorder) {
                    throw std::runtime_error("Nullspace-transformed symbolic plan changed");
                }
                mf::Factor factor;
                factor.threads = threads;
                factor.factorize(transformed, plan);
                const auto x = factor.solve(plan, rhs);
                for (double value : x) {
                    if (std::abs(value - 1.0) > 1e-12) { throw std::runtime_error("Nullspace-transformed solve failed"); }
                }
            }
        }

#ifdef _OPENMP
        // Structural analysis also supports isolated vertices and absent input
        // diagonals; neither symbolic algorithm may invent additional edges.
        {
            mf::Matrix graph;
            graph.n     = 7;
            graph.ptr   = {0, 2, 3, 4, 5, 6, 6, 6};
            graph.row   = {3, 5, 4, 5, 5, 5};
            graph.value.assign(graph.row.size(), 1.0);
            const auto serial   = mf::Symbolic::analyze(graph);
            const auto parallel = mf::Symbolic::analyze(graph, 4);
            if (serial.ptr != parallel.ptr || serial.row != parallel.row || serial.parent != parallel.parent) {
                throw std::runtime_error("Parallel symbolic analysis failed without input diagonals");
            }
            mf::Matrix empty;
            empty.ptr = {0};
            const auto empty_plan = mf::Symbolic::analyze(empty, 4);
            if (empty_plan.ptr != empty.ptr || !empty_plan.row.empty()) {
                throw std::runtime_error("Parallel symbolic analysis failed for an empty matrix");
            }
        }

        // Compare interval counts with independent child-union assembly on
        // irregular patterns, interrupted parent chains and disconnected forests.
        for (int size : {65, 257, 513}) {
            for (int pattern : {0, 1, 2}) {
                mf::Matrix graph;
                graph.n = size;
                graph.ptr.push_back(0);
                std::uint32_t random = static_cast<std::uint32_t>(size + 17 * pattern);
                for (int column = 0; column < size; ++column) {
                    for (int row = column; row < size; ++row) {
                        random = random * 1664525u + 1013904223u;
                        const bool diagonal = row == column && column % 7 != 0;
                        const bool chain    = pattern != 2 && row == column + 1;
                        const bool edge     = row != column && (random % 97u < 4u || chain);
                        const bool separate = pattern == 1 && (row < size / 2) != (column < size / 2);
                        if ((diagonal || edge) && !separate) {
                            graph.row.push_back(row);
                            graph.value.push_back(diagonal ? 10.0 : -0.01);
                        }
                    }
                    graph.ptr.push_back(graph.row.size());
                }
                const auto serial = mf::Symbolic::analyze(graph);
                for (int threads : {2, 4}) {
                    const auto parallel = mf::Symbolic::analyze(graph, threads);
                    if (serial.ptr != parallel.ptr || serial.row != parallel.row ||
                        serial.parent != parallel.parent || serial.first_child != parallel.first_child ||
                        serial.next_child != parallel.next_child || serial.reuse_child != parallel.reuse_child ||
                        serial.max_front != parallel.max_front || serial.factor_flops != parallel.factor_flops ||
                        serial.postorder != parallel.postorder || serial.subtree_begin != parallel.subtree_begin ||
                        serial.subtree_size != parallel.subtree_size || serial.subtree_work != parallel.subtree_work ||
                        serial.roots != parallel.roots) {
                        throw std::runtime_error("Irregular-pattern symbolic mismatch");
                    }
                }
            }
        }

        // Two independent pivots feed a large shared separator. This exercises
        // parallel extend-add rows and the subsequent scalar inherited chain.
        // Scalar mappings cover full spans, short spans with tails, and scatter.
        for (int pattern : {0, 1, 2}) {
            mf::Matrix matrix;
            matrix.n = 1030;
            matrix.ptr.push_back(0);
            for (int column = 0; column < matrix.n; ++column) {
                matrix.row.push_back(column);
                matrix.value.push_back(4.0);
                if (column < 2) {
                    for (int row = 2; row < matrix.n; ++row) {
                        if (column == 0 && pattern == 1 && (row - 2) % 32 >= 16) { continue; }
                        if (column == 0 && pattern == 2 && (row - 2) % 2 != 0) { continue; }
                        matrix.row.push_back(row);
                        matrix.value.push_back(-0.001 * (1.0 + 0.1 * column));
                    }
                }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto symbolic = mf::Symbolic::analyze(matrix);
            mf::Factor serial;
            serial.factorize(matrix, symbolic);
            for (int threads : {2, 4}) {
                mf::Factor parallel;
                parallel.threads = threads;
                parallel.factorize(matrix, symbolic);
                for (std::size_t p = 0; p < serial.value.size(); ++p) {
                    if (parallel.value[p] != serial.value[p]) {
                        throw std::runtime_error("Parallel separator assembly changed a factor entry");
                    }
                }
            }
        }

        // Compare concurrent subtree assembly and substitution on a branching grid
        {
            constexpr int width = 80;
            mf::Matrix matrix;
            matrix.n = width * width;
            matrix.ptr.push_back(0);
            for (int c = 0; c < matrix.n; ++c) {
                matrix.row.push_back(c);
                matrix.value.push_back(5.0);
                if (c % width != width - 1) { matrix.row.push_back(c + 1); matrix.value.push_back(-1.0); }
                if (c + width < matrix.n) { matrix.row.push_back(c + width); matrix.value.push_back(-1.0); }
                matrix.ptr.push_back(matrix.row.size());
            }
            const auto permuted = matrix.permute(mf::nested_dissection(matrix));
            const auto symbolic = mf::Symbolic::analyze(permuted);
            mf::Factor serial;
            serial.factorize(permuted, symbolic);
            std::vector<double> exact(matrix.n);
            for (int i = 0; i < matrix.n; ++i) { exact[i] = std::sin(0.013 * i) + 0.2 * std::cos(0.17 * i); }
            const auto rhs       = permuted.multiply(exact);
            const auto reference = serial.solve(symbolic, rhs);
            for (int threads : {2, 4}) {
                mf::Factor parallel;
                parallel.threads = threads;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    parallel.factorize(permuted, symbolic);
                    if (parallel.inherited_fronts != serial.inherited_fronts) {
                        throw std::runtime_error("Parallel factor missed scalar fronts");
                    }
                    for (std::size_t i = 0; i < serial.value.size(); ++i) {
                        if (std::abs(parallel.value[i] - serial.value[i]) > 1e-12) {
                            throw std::runtime_error("Parallel subtree factor mismatch");
                        }
                    }

                    // Independent subtree solves must merge their shared-ancestor
                    // RHS contributions before entering the separator chain.
                    const auto solution = parallel.solve(symbolic, rhs);
                    for (int i = 0; i < matrix.n; ++i) {
                        if (std::abs(solution[i] - exact[i]) > 1e-11 ||
                            std::abs(solution[i] - reference[i]) > 1e-11) {
                            throw std::runtime_error("Parallel subtree solve mismatch");
                        }
                    }
                }
            }
        }

        // Exercise taskloop row tiles with a large front and an incomplete final panel
        {
            const int n = 1537;
            const int k = 32;
            const int origin = 11;
            const std::size_t full = n + origin;
            std::vector<std::vector<double>> columns(k, std::vector<double>(n));
            std::vector<const double*> pointers;
            for (int p = 0; p < k; ++p) {
                for (int i = 0; i < n; ++i) { columns[p][i] = std::sin(0.013 * i + 0.1 * p); }
                pointers.push_back(columns[p].data());
            }
            std::vector<double> reference(full * (full + 1) / 2, 1.0);
            auto actual = reference;
            std::vector<double> scratch;
            mf::dense::rank_k_subtract(n, k, pointers.data(), reference.data(), origin);
#pragma omp parallel num_threads(4)
            {
#pragma omp single
                mf::dense::rank_k_subtract_packed6(n, k, pointers.data(), actual.data(), origin, scratch);
            }
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::abs(actual[i] - reference[i]) > 1e-11) {
                    throw std::runtime_error("Parallel Schur taskloop mismatch");
                }
            }
        }
#endif

        // Check input triangle conventions, duplicate accumulation and malformed
        // input rejection on small Matrix Market files written next to the binary
        auto write_market = [](const std::string& path, const char* content) {
            std::ofstream file(path, std::ios::binary);
            file << content;
            if (!file) { throw std::runtime_error("Cannot write test matrix: " + path); }
        };
        const std::string directory = std::filesystem::path(argv[0]).parent_path().string();
        const std::string symmetric_path  = directory + "/check_symmetric.mtx";
        const std::string general_path    = directory + "/check_general.mtx";
        const std::string asymmetric_path = directory + "/check_asymmetric.mtx";
        write_market(symmetric_path,
                     "%%MatrixMarket matrix coordinate real symmetric\n"
                     "% Duplicates in a single supplied triangle\n"
                     "3 3 6\n1 1 4\n2 1 0.5\n2 1 0.5\n2 2 3\n3 2 -1\n3 3 2\n");
        write_market(general_path,
                     "%%MatrixMarket matrix coordinate real general\n"
                     "3 3 7\n1 1 4\n2 1 1\n1 2 1\n2 2 3\n3 2 -1\n2 3 -1\n3 3 2\n");
        write_market(asymmetric_path,
                     "%%MatrixMarket matrix coordinate real general\n"
                     "2 2 4\n1 1 4\n2 1 1\n1 2 2\n2 2 3\n");
        const auto imported = mf::Matrix::read_market(symmetric_path);
        const auto general  = mf::Matrix::read_market(general_path);
        if (imported.ptr != general.ptr || imported.row != general.row || imported.value != general.value) {
            throw std::runtime_error("Matrix Market general/symmetric mismatch");
        }
        bool rejected = false;
        try { mf::Matrix::read_market(asymmetric_path); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("Asymmetric input accepted"); }

        // Require an explicit failure rather than modifying an indefinite pivot
        mf::Matrix indefinite;
        indefinite.n     = 2;
        indefinite.ptr   = {0, 2, 3};
        indefinite.row   = {0, 1, 1};
        indefinite.value = {1.0, 2.0, 1.0};
        const auto symbolic = mf::Symbolic::analyze(indefinite);
        mf::Factor factor;
        rejected = false;
        try { factor.factorize(indefinite, symbolic); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("Indefinite matrix accepted"); }
        rejected = false;
        try { factor.solve(symbolic, std::vector<double>(2, 1.0)); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("Solve accepted a partially initialized failed factor"); }
#ifdef _OPENMP
        // A numerical failure must join the team and rethrow, rather than escape a task
        factor.threads = 4;
        rejected = false;
        try { factor.factorize(indefinite, symbolic); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("Parallel indefinite matrix accepted"); }
        if (!factor.value.empty()) { throw std::runtime_error("Failed parallel factor retained numerical entries"); }
#endif
        std::cout << "PASS: dense/2D/3D/disconnected reconstruction, all built orderings, fixed default batches, update blocks 1/8/32/64/128/256, AVX tails/origins, import, SPD rejection\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
