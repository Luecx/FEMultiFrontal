/**
 * @file factor.cpp
 * @brief Implements multi-pivot multifrontal SPD panels and sparse substitution.
 *
 * Consecutive exact-pattern scalar-tree fronts share assembly, mapping and a
 * blocked rectangular triangular solve. The diagonal block uses scalar POTRF.
 * Each amalgamated front owns its children and packed Schur contribution; the
 * existing scalar CSC factor storage is retained. Large SPD substitutions use
 * independent subtrees with private ancestor contributions. This
 * experiment adds neither relaxed fill nor physical DOF grouping, and retains
 * the raw dense kernels and multifrontal contribution-tree architecture.
 */
#include "factor.h"
#include "dense.h"
#include "dense_panel.h"
#include "block_packed_triangle.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf {

/**
 * Dispatch the same multifrontal algorithm with optional phase instrumentation.
 *
 * The disabled specialization contains no phase clock reads. The profiled path
 * accumulates nonoverlapping wall-clock segments, then assigns unclassified
 * traversal and final temporary destruction to Other. This reports measurement
 * overhead rather than subtracting an assumed timer cost from the result.
 */
void Factor::factorize(const Matrix& matrix, const Symbolic& symbolic) {
    // Validate execution settings before entering any OpenMP tasks
    if (threads < 1) { throw std::runtime_error("Thread count must be positive"); }
    if (threads > 1 && profile_enabled) {
        throw std::runtime_error("Phase profiling requires one thread; parallel intervals overlap");
    }
#ifndef _OPENMP
    if (threads > 1) { throw std::runtime_error("Build with WITH_OPENMP=1 for multiple threads"); }
#endif

    // Reset previous diagnostics without introducing clocks in the normal path
    phase_seconds.fill(0.0);
    threads_used = 1;
    try {
#ifdef _OPENMP
        if (threads > 1) {
            factorize_impl<false, true>(matrix, symbolic);
            return;
        }
#endif
        if (!profile_enabled) {
            factorize_impl<false, false>(matrix, symbolic);
            return;
        }

        // Include final local-container destruction in the profiled total
        using Clock = std::chrono::steady_clock;
        const auto begin = Clock::now();
        factorize_impl<true, false>(matrix, symbolic);
        const double total = std::chrono::duration<double>(Clock::now() - begin).count();
        const double measured = std::accumulate(phase_seconds.begin(), phase_seconds.end(), 0.0);
        phase_seconds[Other] += std::max(0.0, total - measured);
    } catch (...) {
        // Reject subsequent solves after a failed factorization. Some columns
        // have not been initialized yet; none may be exposed as a valid factor.
        value.clear();
        throw;
    }
}

/**
 * Assemble and eliminate exact amalgamated fronts in child-before-parent order.
 *
 * Exact scalar-tree chains are grouped up to the requested pivot width.
 * Original A entries and external child contributions are assembled once per
 * exact group. A row-wise transpose fills its CSC pivot panel. Dense POTRF
 * handles the leading block; register-blocked TRSM solves the rectangular
 * suffix, followed by one rank-width Schur update. Groups are bounded by the
 * same batch size as the scalar implementation, so no deferred columns cross
 * group boundaries. Persistent factor columns and the contribution tree remain
 * unchanged; this is multifrontal assembly rather than global sparse updates.
 * Nonpositive or nonfinite pivots terminate factorization with their index.
 *
 * @tparam Profile Enable wall-clock boundaries without changing arithmetic.
 */
template<bool Profile, bool Parallel>
void Factor::factorize_impl(const Matrix& matrix, const Symbolic& symbolic) {
    // Compile phase boundaries out of the unprofiled specialization
    using Clock = std::chrono::steady_clock;
    Clock::time_point stamp{};
    if constexpr (Profile) { stamp = Clock::now(); }
    auto finish_phase = [&](ProfilePhase phase) {
        if constexpr (Profile) {
            const auto next = Clock::now();
            phase_seconds[phase] += std::chrono::duration<double>(next - stamp).count();
            stamp = next;
        }
    };

    // Validate the only numerical tuning parameter before allocating buffers
    if (update_block_size < 1 || update_block_size > 256) {
        throw std::runtime_error("Schur update block size must be in [1, 256]");
    }
    // Build exact chain groups from scalar symbolic index-set matches. A merge
    // crosses no branch: the parent must have this child as its only child and
    // exactly the child's trailing pattern. Group pivots are a prefix of the
    // first front; no fill is added and no numerical values enter the decision.
    const int n = matrix.n;
    std::vector<int> front_first(n, -1);
    std::vector<int> front_width(n, 0);
    assembled_fronts = 0;
    merged_pivots    = 0;
    for (int first = 0; first < n; ++first) {
        if (front_first[first] != -1) { continue; }
        int head  = first;
        int width = 1;
        front_first[head] = first;
        while (width < update_block_size) {
            const int parent = symbolic.parent[head];
            if (parent == -1 || symbolic.reuse_child[parent] != head) { break; }
            head = parent;
            front_first[head] = first;
            ++width;
        }
        front_width[head] = width;
        ++assembled_fronts;
        merged_pivots += width - 1;
    }
    finish_phase(Other);

    // Prepare persistent factors and transient dense contribution ownership
    value.resize(symbolic.row.size());
    huge_page_hints_accepted = prefer_huge_pages && memory::advise_huge_pages(value.data(), value.size() * sizeof(double));
    std::vector<dense::BlockPackedTriangle> contributions(n);
    std::vector<int> origin(n, 0);
    std::vector<memory::Indices> worker_position(Parallel ? threads : 1);
    std::size_t live_bytes = 0;
    std::atomic<std::size_t> parallel_live{0};
    std::atomic<std::size_t> parallel_peak{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    peak_contribution_bytes = 0;
    minimum_pivot           = std::numeric_limits<double>::infinity();
    inherited_fronts        = 0;
    finish_phase(Memory);

    // Count actual live contribution payloads across independent subtree tasks
    auto allocate_bytes = [&](std::size_t bytes) {
        if constexpr (Parallel) {
            const std::size_t live = parallel_live.fetch_add(bytes, std::memory_order_relaxed) + bytes;
            std::size_t peak = parallel_peak.load(std::memory_order_relaxed);
            while (peak < live && !parallel_peak.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {}
        } else {
            live_bytes += bytes;
            peak_contribution_bytes = std::max(peak_contribution_bytes, live_bytes);
        }
    };
    auto release_bytes = [&](std::size_t bytes) {
        if constexpr (Parallel) {
            parallel_live.fetch_sub(bytes, std::memory_order_relaxed);
        } else {
            live_bytes -= bytes;
        }
    };

    // Reuse the global-index map per worker. Every entry queried for a pivot is
    // rebuilt before use, and no map entry is read across a task scheduling point.
    // Child maps, suffix pointers and dense panels do survive taskloop waits and
    // therefore remain private to the suspended numerical task.
    auto run_nodes = [&](const int* nodes, int count) {
        int worker = 0;
#ifdef _OPENMP
        if constexpr (Parallel) { worker = omp_get_thread_num(); }
#endif
        auto& position = worker_position[worker];
        if (position.empty()) { position.assign(n, -1); }
        memory::Indices child_map;
        std::vector<int> child_run_ends;
        std::vector<double*> panel_columns;
        std::vector<const double*> update_columns;
        std::vector<double> schur_scratch;
        panel_columns.reserve(update_block_size);
        double inverse[256];
        update_columns.reserve(update_block_size);
        double task_minimum = std::numeric_limits<double>::infinity();
        std::size_t task_inherited = 0;
        std::size_t task_huge_hints = prefer_huge_pages && memory::advise_huge_pages(position.data(), position.size() * sizeof(int));
        finish_phase(Memory);

        // Only group heads own numerical work; absorbed scalar nodes are skipped
        for (int index = 0; index < count; ++index) {
            if constexpr (Parallel) { if (failed.load(std::memory_order_relaxed)) { return; } }
            const int pivot = nodes ? nodes[index] : index;
            const int width = front_width[pivot];
            if (width == 0) { continue; }
            const int         first   = front_first[pivot];
            const std::size_t begin   = symbolic.ptr[first];
            const int         size    = static_cast<int>(symbolic.ptr[first + 1] - begin);
            const int         updates = size - 1;
            double*           column  = value.data() + begin;

            // One global-to-local mapping serves all pivots of this front.
            // Initialize every original column before any child/dense taskloop
            // can lend this worker and overwrite its shared position scratch.
            for (int i = 0; i < size; ++i) {
                position[symbolic.row[begin + i]] = i;
            }
            panel_columns.clear();
            for (int local = 0; local < width; ++local) {
                const int scalar = symbolic.row[begin + local];
                double* target = value.data() + symbolic.ptr[scalar];
                panel_columns.push_back(target);
                std::fill_n(target, size - local, 0.0);
                for (std::size_t p = matrix.ptr[scalar]; p < matrix.ptr[scalar + 1]; ++p) {
                    target[position[matrix.row[p]] - local] = matrix.value[p];
                }
            }
            auto& update = contributions[pivot];
            const int  only_child = symbolic.reuse_child[first];
            const bool inherit    = only_child != -1;
            finish_phase(Assembly);

            // Reuse a matching child's packed storage instead of copying its trailing matrix.
            // Eliminated leading rows stay allocated; origin exposes the remaining subtriangle.
            if (inherit) {
                const int child_origin = origin[only_child];
                update = std::move(contributions[only_child]);
                finish_phase(Memory);
                for (int i = 0; i < size; ++i) {
                    const std::size_t r = static_cast<std::size_t>(child_origin + i);
                    column[i] += update.at(static_cast<int>(r), child_origin);
                }
                origin[pivot] = child_origin + 1;
                ++task_inherited;
                finish_phase(Assembly);
            } else {
                update.assign_triangle(updates);
                allocate_bytes(update.size() * sizeof(double));
                if (prefer_huge_pages && memory::advise_huge_pages(update.data(), update.size() * sizeof(double))) {
                    ++task_huge_hints;
                }
                finish_phase(Memory);
            }

            // Extend-add all child Schur complements into this scalar front
            for (int child = symbolic.first_child[first]; child != -1; child = symbolic.next_child[child]) {
                if (inherit) {
                    break;
                }
                const std::size_t child_begin = symbolic.ptr[child] + 1;
                const int child_size = static_cast<int>(symbolic.ptr[child + 1] - child_begin);
                // A previous child's taskloop may have lent this worker to a
                // different numerical task. Restore the map before reading it.
                if constexpr (Parallel) {
                    if (child != symbolic.first_child[first]) {
                        for (int i = 0; i < size; ++i) { position[symbolic.row[begin + i]] = i; }
                    }
                }
                child_map.resize(child_size);
                for (int i = 0; i < child_size; ++i) {
                    child_map[i] = position[symbolic.row[child_begin + i]];
                }

                // Block-packed Extend-add: source index set is sorted but
                // mapping into parent can have gaps.
                const auto& source = contributions[child];
                const int child_origin = origin[child];
                auto assemble_rows = [&](int first, int last) {
                    for (int i = first; i < last; ++i) {
                        const int r = child_map[i];
                        column[r] += source.at(child_origin + i, child_origin);
                        for (int j = 1; j <= i; ++j) {
                            update.at(r - 1, child_map[j] - 1) +=
                                source.at(child_origin + i, child_origin + j);
                        }
                    }
                };
                if constexpr (Parallel) {
#ifdef _OPENMP
                    if (child_size >= 1024) {
                        const int tiles = (child_size + 31) / 32;
#pragma omp taskloop grainsize(1) shared(assemble_rows)
                        for (int tile = 0; tile < tiles; ++tile) {
                            assemble_rows(tile * 32, std::min(child_size, (tile + 1) * 32));
                        }
                    } else { assemble_rows(0, child_size); }
#endif
                } else {
                    assemble_rows(0, child_size);
                }
                finish_phase(Assembly);

                // Child storage has no further consumers after its parent assembles it
                release_bytes(source.size() * sizeof(double));
                contributions[child].clear();
                finish_phase(Memory);
            }
            finish_phase(Other);

            // Read the group's contribution columns together. Row-wise tiles
            // turn repeated strided gathers into contiguous lower-triangle reads.
            dense::add_block_packed_panel(size, width, panel_columns.data(), update, origin[pivot]);
            finish_phase(Assembly);

            // Factor only the dense leading block, then solve its rectangular
            // suffix as multiple right-hand sides with register-blocked TRSM.
            double panel_minimum = std::numeric_limits<double>::infinity();
            const int bad = dense::factor_pivot_block(width, panel_columns.data(), inverse, panel_minimum);
            if (bad != -1) {
                throw std::runtime_error("Nonpositive/nonfinite SPD pivot at permuted index " +
                                         std::to_string(symbolic.row[begin + bad]));
            }
            task_minimum = std::min(task_minimum, panel_minimum);
            finish_phase(Elimination);
            dense::solve_panel(size,width,panel_columns.data(),inverse);
            finish_phase(PivotCorrection);
            task_inherited += width - 1;
            origin[pivot] += width - 1;

            // A whole group exports one fully updated Schur complement. Groups
            // end exactly at a batch limit or an index-set/branch boundary, so
            // no deferred scalar columns survive across group ownership.
            const int trailing = size - width;
            update_columns.clear();
            for (int local = 0; local < width; ++local) {
                update_columns.push_back(panel_columns[local] + width - local);
            }
            dense::rank_k_subtract_bpt(trailing, width, update_columns.data(),
                                       update, origin[pivot]);
            finish_phase(SchurUpdate);
            if (symbolic.parent[pivot] == -1) {
                release_bytes(update.size() * sizeof(double));
                update.clear();
                finish_phase(Memory);
            }
            finish_phase(Other);
        }

        // Combine only small task summaries; factor columns and child buffers need no lock
        if constexpr (Parallel) {
#ifdef _OPENMP
#pragma omp critical(mf_factor_statistics)
#endif
            {
                minimum_pivot    = std::min(minimum_pivot, task_minimum);
                inherited_fronts += task_inherited;
                huge_page_hints_accepted += task_huge_hints;
            }
        } else {
            minimum_pivot    = task_minimum;
            inherited_fronts = task_inherited;
            huge_page_hints_accepted += task_huge_hints;
        }
    };

    if constexpr (!Parallel) {
        // Preserve the original increasing-column execution in the one-thread path
        run_nodes(nullptr, n);
    } else {
#ifdef _OPENMP
        // Coarsen subtrees; exact groups never cross these branching boundaries.
        // Workspaces are allocated only when a task starts its own numerical work.
        std::function<void(int, int)> visit;
        visit = [&](int root, int depth) {
            if (failed.load(std::memory_order_relaxed)) { return; }
            try {
                if (symbolic.subtree_work[root] <= 2e7 || symbolic.subtree_size[root] <= 256 || depth >= 64) {
                    run_nodes(symbolic.postorder.data() + symbolic.subtree_begin[root], symbolic.subtree_size[root]);
                    return;
                }
                std::vector<int> chain;
                int head = root;
                chain.push_back(head);
                while (symbolic.first_child[head] != -1 &&
                       symbolic.next_child[symbolic.first_child[head]] == -1) {
                    head = symbolic.first_child[head];
                    chain.push_back(head);
                }
#pragma omp taskgroup
                {
                    for (int child = symbolic.first_child[head]; child != -1; child = symbolic.next_child[child]) {
#pragma omp task firstprivate(child, depth) shared(visit)
                        visit(child, depth + 1);
                    }
                }
                if (failed.load(std::memory_order_relaxed)) { return; }
                std::reverse(chain.begin(), chain.end());
                run_nodes(chain.data(), static_cast<int>(chain.size()));
            } catch (...) {
                // Exceptions cannot cross an OpenMP task boundary; rethrow after the team joins
#pragma omp critical(mf_factor_failure)
                {
                    if (!failure) { failure = std::current_exception(); }
                }
                failed.store(true, std::memory_order_relaxed);
            }
        };
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            {
                threads_used = omp_get_num_threads();
#pragma omp taskgroup
                {
                    for (int root : symbolic.roots) {
#pragma omp task firstprivate(root) shared(visit)
                        visit(root, 0);
                    }
                }
            }
        }
        if (failure) { std::rethrow_exception(failure); }
        peak_contribution_bytes = parallel_peak.load(std::memory_order_relaxed);
#endif
    }
}

/**
 * Solve L y = b followed by L^T x = y in the permuted coordinate system.
 *
 * The forward pass subtracts normalized pivots from higher-index ancestor rows.
 * Exact consecutive chains share a row list and update each suffix entry once
 * per group of at most eight columns, limiting simultaneous factor streams.
 * The reverse pass gathers the transpose contribution from those same rows.
 * Neither pass reads dense contribution storage or the original matrix.
 * Large single-root plans split into independent subtrees. Forward tasks use
 * private RHS contributions to avoid shared-ancestor scatter races; transpose
 * tasks start only after the common separator has been solved. Serial and
 * parallel paths use the same scalar CSC factors.
 */
std::vector<double> Factor::solve(const Symbolic& symbolic, const std::vector<double>& rhs) const {
    // Validate factor and right-hand-side dimensions before substitution
    const int n = static_cast<int>(symbolic.parent.size());
    if (rhs.size() != static_cast<std::size_t>(n) || value.size() != symbolic.row.size()) {
        throw std::runtime_error("Factor/solve dimension mismatch");
    }
    auto x = rhs;

    // Exact consecutive scalar chains share one row list. Sparse/small factors
    // keep the legacy traversal so plan construction cannot dominate a solve.
    constexpr int panel_limit = 8;
    std::vector<unsigned char> panel_width;
    if (n >= 1024 && value.size() >= static_cast<std::size_t>(n) * 16) {
        panel_width.assign(n, 0);
        for (int first = 0; first < n;) {
            int head  = first;
            int width = 1;
            while (width < panel_limit && head + 1 < n &&
                   symbolic.parent[head] == head + 1 && symbolic.reuse_child[head + 1] == head) {
                ++head;
                ++width;
            }
            panel_width[head] = static_cast<unsigned char>(width);
            first = head + 1;
        }
    }

    // Apply one scalar column. Off-diagonal rows are ancestors in the exact
    // elimination tree, so independent subtrees cannot update one another.
    auto forward = [&](int c, double* work) {
        const std::size_t begin = symbolic.ptr[c];
        work[c] /= value[begin];
        const double pivot = work[c];
        for (std::size_t p = begin + 1; p < symbolic.ptr[c + 1]; ++p) {
            work[symbolic.row[p]] -= value[p] * pivot;
        }
    };
    // Add private RHS entries at the same point as the scalar algorithm. A
    // group cannot cross a frontier branch because every link has one child.
    auto forward_group = [&](int head, double* work, const double* input) {
        const int width = panel_width[head];
        if (width == 0) { return; }
        if (width == 1) {
            if (input) { work[head] += input[head]; }
            forward(head, work);
            return;
        }
        const int first = head - width + 1;
        const double* columns[panel_limit];
        for (int j = 0; j < width; ++j) { columns[j] = value.data() + symbolic.ptr[first + j]; }
        const std::size_t begin = symbolic.ptr[first];
        const int         size  = static_cast<int>(symbolic.ptr[first + 1] - begin);
        dense::forward_panel(size, width, symbolic.row.data() + begin, columns, work, input);
    };
    auto backward = [&](int c) {
        const std::size_t begin   = symbolic.ptr[c];
        const int         entries = static_cast<int>(symbolic.ptr[c + 1] - begin - 1);
        const double      sum     = dense::indexed_dot(
            entries,
            symbolic.row.data() + begin + 1,
            value.data() + begin + 1,
            x.data()
        );
        x[c] = (x[c] - sum) / value[begin];
    };

#ifdef _OPENMP
    // Split only large, connected plans. Small or highly branching forests
    // retain the allocation-free serial traversal below.
    if (threads > 1 && n >= 1024 && value.size() >= 100000 && symbolic.roots.size() == 1) {
        const int workers = std::min({threads, n, omp_get_thread_limit()});
        std::vector<int> frontier{symbolic.roots.front()};
        std::vector<int> separator;
        while (static_cast<int>(frontier.size()) < workers) {
            // Expand the largest admissible branch, keeping at most one RHS
            // buffer per worker. Unary ancestor chains join the separator.
            int chosen = -1;
            int head   = -1;
            int weight = 0;
            for (int i = 0; i < static_cast<int>(frontier.size()); ++i) {
                int branch = frontier[i];
                while (symbolic.first_child[branch] != -1 &&
                       symbolic.next_child[symbolic.first_child[branch]] == -1) {
                    branch = symbolic.first_child[branch];
                }
                int children = 0;
                for (int c = symbolic.first_child[branch]; c != -1; c = symbolic.next_child[c]) { ++children; }
                if (children >= 2 && static_cast<int>(frontier.size()) - 1 + children <= workers &&
                    symbolic.subtree_size[branch] > weight) {
                    chosen = i;
                    head   = branch;
                    weight = symbolic.subtree_size[branch];
                }
            }
            if (chosen == -1) { break; }
            for (int node = frontier[chosen]; node != head; node = symbolic.first_child[node]) {
                separator.push_back(node);
            }
            separator.push_back(head);
            frontier.erase(frontier.begin() + chosen);
            for (int c = symbolic.first_child[head]; c != -1; c = symbolic.next_child[c]) { frontier.push_back(c); }
        }

        if (frontier.size() > 1) {
            // Allocate before entering OpenMP so allocation failures propagate
            // normally; each worker initializes its own contribution buffer.
            std::vector<memory::UninitializedValues> contributions(frontier.size());
            for (auto& buffer : contributions) { buffer.resize(n); }
            std::sort(separator.begin(), separator.end());
            const int tasks = static_cast<int>(frontier.size());
#pragma omp parallel for num_threads(workers) schedule(static)
            for (int task = 0; task < tasks; ++task) {
                auto& work = contributions[task];
                std::fill(work.begin(), work.end(), 0.0);
                const int root  = frontier[task];
                const int first = symbolic.subtree_begin[root];
                const int last  = first + symbolic.subtree_size[root];
                for (int index = first; index < last; ++index) {
                    const int node = symbolic.postorder[index];
                    if (panel_width.empty()) {
                        work[node] += rhs[node];
                        forward(node, work.data());
                        x[node] = work[node];
                    } else {
                        const int width = panel_width[node];
                        if (width == 0) { continue; }
                        forward_group(node, work.data(), rhs.data());
                        for (int c = node - width + 1; c <= node; ++c) { x[c] = work[c]; }
                    }
                }
            }

            // Merge only ancestor contributions, in a fixed order, then solve
            // the remaining separator after all child subtrees are complete.
            for (int node : separator) {
                for (int task = 0; task < tasks; ++task) { x[node] += contributions[task][node]; }
            }
            for (int node : separator) {
                if (panel_width.empty()) { forward(node, x.data()); }
                else { forward_group(node, x.data(), nullptr); }
            }

            // Transpose substitution reverses dependencies: ancestors first,
            // then subtrees whose unknowns and written solution entries differ.
            for (auto it = separator.rbegin(); it != separator.rend(); ++it) { backward(*it); }
#pragma omp parallel for num_threads(workers) schedule(static)
            for (int task = 0; task < tasks; ++task) {
                const int root  = frontier[task];
                const int first = symbolic.subtree_begin[root];
                const int last  = first + symbolic.subtree_size[root];
                for (int index = last - 1; index >= first; --index) { backward(symbolic.postorder[index]); }
            }
            return x;
        }
    }
#endif

    // Apply groups in pivot order; small or sparse plans retain scalar traversal
    for (int c = 0; c < n; ++c) {
        if (panel_width.empty()) { forward(c, x.data()); }
        else { forward_group(c, x.data(), nullptr); }
    }
    for (int c = n - 1; c >= 0; --c) { backward(c); }
    return x;
}

} // namespace mf
