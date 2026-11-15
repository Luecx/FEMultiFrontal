/**
 * @file symbolic.cpp
 * @brief Builds exact scalar frontal index sets through elimination-tree assembly.
 *
 * Original lower-column rows and child update rows form each front. The first
 * update index becomes its parent pivot. This mirrors numerical extend-add and
 * avoids explicit clique insertion into a mutable elimination graph.
 */
#include "symbolic.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf {

/**
 * Analyze a permuted symmetric matrix without inspecting its numerical values.
 *
 * The serial path merges sorted child suffixes with original columns. The
 * parallel path first computes the elimination tree from original upper
 * adjacency, then reaches that immutable tree independently from each matrix
 * row. Consecutive parent chains turn count updates into interval endpoints;
 * prefix sums recover exact column counts and reserve disjoint worker ranges.
 * A second tree-reach pass writes factor rows without atomics, with each worker
 * also initializing the diagonal entries in its own row range. Partitions use
 * graph predecessor path lengths as work estimates rather than equal row counts.
 * The same boundaries in both passes guarantee ascending factor rows even when
 * workers finish out of order; no physical variable blocks are required.
 * Factor storage includes pivots with structurally absent input diagonals.
 */
Symbolic Symbolic::analyze(const Matrix& matrix, int threads) {
    // Validate execution mode before constructing the immutable scalar plan
    if (threads < 1) { throw std::runtime_error("Symbolic thread count must be positive"); }
#ifndef _OPENMP
    if (threads > 1) { throw std::runtime_error("Build with WITH_OPENMP=1 for parallel analysis"); }
#endif
    // Initialize persistent compressed factor rows and tree links
    const int n = matrix.n;
    Symbolic result;
    result.ptr.reserve(static_cast<std::size_t>(n) + 1);
    result.ptr.push_back(0);
    result.parent.assign(n, -1);
    result.first_child.assign(n, -1);
    result.next_child.assign(n, -1);
    if (n == 0) { return result; }
    std::vector<int> marker(n, -1);
    std::vector<int> front;

    // Assemble each front's structural union before assigning its parent
    if (threads == 1) {
        for (int pivot = 0; pivot < n; ++pivot) {
            front.clear();
            // Begin with a sorted child suffix. Along an inherited chain, all
            // original entries are already present and no sorting is necessary.
            const int first_child = result.first_child[pivot];
            if (first_child != -1) {
                front.insert(front.end(), result.row.begin() + result.ptr[first_child] + 1,
                             result.row.begin() + result.ptr[first_child + 1]);
                for (int row : front) { marker[row] = pivot; }
            }
            const auto sorted_size = front.size();
            auto insert = [&](int row) {
                if (marker[row] != pivot) {
                    marker[row] = pivot;
                    front.push_back(row);
                }
            };
            insert(pivot);
            for (std::size_t p = matrix.ptr[pivot]; p < matrix.ptr[pivot + 1]; ++p) {
                insert(matrix.row[p]);
            }
            for (int child = result.first_child[pivot]; child != -1; child = result.next_child[child]) {
                if (child == first_child) { continue; }
                for (std::size_t p = result.ptr[child] + 1; p < result.ptr[child + 1]; ++p) {
                    insert(result.row[p]);
                }
            }
            if (front.size() != sorted_size) { std::sort(front.begin(), front.end()); }
            result.row.insert(result.row.end(), front.begin(), front.end());
            result.ptr.push_back(result.row.size());

            // Attach this scalar front to the pivot that first consumes its contribution
            if (front.size() > 1) {
                const int parent = front[1];
                result.parent[pivot]     = parent;
                result.next_child[pivot] = result.first_child[parent];
                result.first_child[parent] = pivot;
            }
            const double updates = static_cast<double>(front.size() - 1);
            result.max_front     = std::max(result.max_front, front.size());
            result.factor_flops += 1.0 + updates + updates * (updates + 1.0);
        }
    } else {
#ifdef _OPENMP
        // Transpose structural lower columns into original row predecessors
        std::vector<std::size_t> row_ptr(static_cast<std::size_t>(n) + 1, 0);
        for (int row : matrix.row) { ++row_ptr[row + 1]; }
        std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        auto row_next = row_ptr;
        std::vector<int> predecessors(matrix.row.size());
        for (int column = 0; column < n; ++column) {
            for (std::size_t p = matrix.ptr[column]; p < matrix.ptr[column + 1]; ++p) {
                predecessors[row_next[matrix.row[p]]++] = column;
            }
        }

        // Path compression computes the scalar elimination tree without fill
        std::vector<int> ancestor(n, -1);
        for (int pivot = 0; pivot < n; ++pivot) {
            for (std::size_t p = row_ptr[pivot]; p < row_ptr[pivot + 1]; ++p) {
                int node = predecessors[p];
                while (node != -1 && node < pivot) {
                    const int next = ancestor[node];
                    ancestor[node] = pivot;
                    if (next == -1) { result.parent[node] = pivot; }
                    node = next;
                }
            }
        }

        const int workers = std::min(threads, std::max(1, n));

        // Reuse the completed path-compression workspace for tree depths.
        // Every parent has a larger index, so descending order visits it first.
        for (int pivot = n - 1; pivot >= 0; --pivot) {
            const int parent = result.parent[pivot];
            ancestor[pivot] = parent == -1 ? 1 : ancestor[parent] + 1;
        }

        // Reaching a predecessor follows its tree path up to the current row.
        // Summed path lengths overestimate shared paths, but predict separator
        // work much better than row counts without constructing the fill first.
        std::vector<double> work_prefix(static_cast<std::size_t>(n) + 1, 1.0);
        work_prefix[0] = 0.0;
        for (int column = 0; column < n; ++column) {
            for (std::size_t p = matrix.ptr[column]; p < matrix.ptr[column + 1]; ++p) {
                const int row = matrix.row[p];
                if (row != column) { work_prefix[row + 1] += std::max(1, ancestor[column] - ancestor[row]); }
            }
        }
        std::partial_sum(work_prefix.begin(), work_prefix.end(), work_prefix.begin());
        std::vector<int> partition(static_cast<std::size_t>(workers) + 1, 0);
        partition[workers] = n;
        for (int worker = 1; worker < workers; ++worker) {
            const double target = work_prefix.back() * worker / workers;
            partition[worker] = static_cast<int>(std::lower_bound(work_prefix.begin(), work_prefix.end(), target) - work_prefix.begin());
        }

        // Compress consecutive parent links into intervals. The depth workspace
        // is no longer needed after partitioning. ancestor[v] now denotes the
        // exclusive end of the chain v, v+1, ... with parent[j] = j+1.
        for (int node = n - 1; node >= 0; --node) {
            ancestor[node] = result.parent[node] == node + 1 ? ancestor[node + 1] : node + 1;
        }

        // Each row reaches a suffix of every encountered chain. Remember its
        // lowest visited node; extending that suffix contributes one interval
        // to the column counts instead of incrementing every ancestor separately.
        const std::size_t workspace_size = static_cast<std::size_t>(workers) * n;
        std::vector<int> markers(workspace_size, -1);
        std::vector<int> lowest(workspace_size);
        std::vector<int> counts(workspace_size, 0);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int worker = 0; worker < workers; ++worker) {
            int* seen  = markers.data() + static_cast<std::size_t>(worker) * n;
            int* low   = lowest.data() + static_cast<std::size_t>(worker) * n;
            int* count = counts.data() + static_cast<std::size_t>(worker) * n;
            const int first = partition[worker];
            const int last  = partition[worker + 1];
            for (int pivot = first; pivot < last; ++pivot) {
                for (std::size_t p = row_ptr[pivot]; p < row_ptr[pivot + 1]; ++p) {
                    int node = predecessors[p];
                    while (node != -1 && node < pivot) {
                        const int end   = ancestor[node];
                        const int chain = end - 1;
                        int limit = std::min(end, pivot);
                        if (seen[chain] == pivot) {
                            limit = std::min(limit, low[chain]);
                            if (node >= limit) { break; }
                        }
                        seen[chain] = pivot;
                        low[chain]  = node;
                        ++count[node];
                        --count[limit];
                        if (limit != end) { break; }
                        node = result.parent[chain];
                    }
                }
            }

            // Convert interval endpoints to exact per-column counts. Every
            // interval is bounded by its row pivot, so the diagonal is excluded.
            std::partial_sum(count, count + n, count);
        }

        // Reserve exact per-worker column ranges, with the diagonal first
        result.ptr.resize(static_cast<std::size_t>(n) + 1);
        std::vector<std::size_t> cursor(workspace_size);
        std::size_t entries = 0;
        for (int column = 0; column < n; ++column) {
            result.ptr[column] = entries++;
            for (int worker = 0; worker < workers; ++worker) {
                const std::size_t index = static_cast<std::size_t>(worker) * n + column;
                cursor[index] = entries;
                entries += static_cast<std::size_t>(counts[index]);
            }
            const double updates = static_cast<double>(entries - result.ptr[column] - 1);
            result.max_front = std::max(result.max_front, entries - result.ptr[column]);
            result.factor_flops += 1.0 + updates + updates * (updates + 1.0);
        }
        result.ptr[n] = entries;
        result.row.resize(entries);

        // Each partition writes increasing row IDs to its exclusive column ranges
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int worker = 0; worker < workers; ++worker) {
            int* seen = markers.data() + static_cast<std::size_t>(worker) * n;
            auto* next = cursor.data() + static_cast<std::size_t>(worker) * n;
            std::fill_n(seen, n, -1);
            const int first = partition[worker];
            const int last  = partition[worker + 1];
            for (int pivot = first; pivot < last; ++pivot) {
                // The row owner also initializes its diagonal. No serial pass
                // first-touches the entire large factor-index allocation.
                result.row[result.ptr[pivot]] = pivot;
                seen[pivot] = pivot;
                for (std::size_t p = row_ptr[pivot]; p < row_ptr[pivot + 1]; ++p) {
                    int node = predecessors[p];
                    while (node != -1 && seen[node] != pivot) {
                        seen[node] = pivot;
                        result.row[next[node]++] = pivot;
                        node = result.parent[node];
                    }
                }
            }
        }

        // Recreate the same child-list order as ascending scalar-front assembly
        for (int pivot = 0; pivot < n; ++pivot) {
            const int parent = result.parent[pivot];
            if (parent != -1) {
                result.next_child[pivot] = result.first_child[parent];
                result.first_child[parent] = pivot;
            }
        }
#endif
    }

    // Identify exact trailing-index matches once, independently of numeric values.
    // Matching fronts may inherit storage while still eliminating one scalar each.
    // Exact assembly includes every child suffix in its parent front, so equal
    // cardinalities prove equality of these sorted sets without rereading them.
    result.reuse_child.assign(n, -1);
    for (int pivot = 0; pivot < n; ++pivot) {
        const int child = result.first_child[pivot];
        if (child != -1 && result.next_child[child] == -1 &&
            result.ptr[child + 1] - result.ptr[child] - 1 == result.ptr[pivot + 1] - result.ptr[pivot]) {
            result.reuse_child[pivot] = child;
        }
    }

    // Accumulate work estimates without changing the scalar elimination forest
    result.subtree_begin.resize(n);
    result.subtree_size.assign(n, 1);
    result.subtree_work.resize(n);
    for (int pivot = 0; pivot < n; ++pivot) {
        const double updates = static_cast<double>(result.ptr[pivot + 1] - result.ptr[pivot] - 1);
        result.subtree_work[pivot] = 1.0 + updates + updates * (updates + 1.0);
    }
    for (int pivot = 0; pivot < n; ++pivot) {
        const int parent = result.parent[pivot];
        if (parent == -1) {
            result.roots.push_back(pivot);
        } else {
            result.subtree_size[parent] += result.subtree_size[pivot];
            result.subtree_work[parent] += result.subtree_work[pivot];
        }
    }

    // Construct subtree-contiguous postorder iteratively, avoiding deep call stacks
    result.postorder.reserve(n);
    auto next_child = result.first_child;
    std::vector<int> stack;
    for (int root : result.roots) {
        stack.push_back(root);
        while (!stack.empty()) {
            const int pivot = stack.back();
            const int child = next_child[pivot];
            if (child != -1) {
                next_child[pivot] = result.next_child[child];
                stack.push_back(child);
            } else {
                result.postorder.push_back(pivot);
                result.subtree_begin[pivot] = static_cast<int>(result.postorder.size()) - result.subtree_size[pivot];
                stack.pop_back();
            }
        }
    }
    return result;
}

} // namespace mf
