/**
 * @file blocked_ldlt.cpp
 * @brief True multifrontal, supernode-amalgamated symmetric indefinite LDLT.
 *
 * Threshold-selected 1x1/2x2 pivots remain exact front-local operations.
 * Only columns needed by the next pivot are updated eagerly; the full trailing
 * Schur complement is updated by a cache-blocked rank-k operation at panel
 * boundaries. Unstable fully-summed candidates are passed to parent fronts.
 * There is no Cholesky, diagonal shift, or positive-definiteness shortcut.
 */
#include "blocked_ldlt.h"
#include "pivot_factor.h"
#include "memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf {
namespace {

// A transient Schur contribution with one consuming parent. Its allocated
// packed triangle may be inherited without copying along an exact tree link.
struct BlockFront {
    std::vector<int> rows;
    memory::Values values;
    int origin = 0;
    int stride = 0;
};

// Each accepted scalar column of a 1x1/2x2 pivot is stored independently.
// left and right are rank-major arrays with physical stride dim. Their product
// reconstructs the Schur correction in the accepted pivot order.
struct PivotPanel {
    int dim = 0;
    int rank = 0;
    std::vector<double> left;
    std::vector<double> right;

    explicit PivotPanel(int width, int capacity) : dim(width) {
        left.reserve(static_cast<std::size_t>(width) * capacity);
        right.reserve(static_cast<std::size_t>(width) * capacity);
    }
    void append(int count) {
        const std::size_t size = static_cast<std::size_t>(rank + count) * dim;
        left.resize(size, 0.0);
        right.resize(size, 0.0);
    }
    void clear() {
        rank = 0;
        left.clear();
        right.clear();
    }
};

// Consume an entire accepted pivot panel while streaming each lower-triangle
// output tile only once. Holes from noncontiguous pivoting remain supported.
// The scalar path preserves the pivot acceptance order for all corrections.
void rank_k_update(int origin, int dim, const std::vector<int>& active,
                   const PivotPanel& panel, double* values) {
    if (panel.rank == 0) { return; }
    for (std::size_t i = 0; i < active.size(); ++i) {
        const int r = active[i];
        const std::size_t physical = static_cast<std::size_t>(origin + r);
        double* const target = values + physical * (physical + 1) / 2 + origin;
        auto update_scalar = [&](std::size_t index) {
            const int c = active[index];
            double v = target[c];
            for (int p = 0; p < panel.rank; ++p) {
                const std::size_t base = static_cast<std::size_t>(p) * dim;
                v -= panel.left[base + r] * panel.right[base + c];
            }
            if (!std::isfinite(v)) {
                throw std::runtime_error("Nonfinite blocked LDLT Schur update");
            }
            target[c] = v;
        };
        std::size_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
        const __m256d sign = _mm256_set1_pd(-0.0);
        const __m256d limit = _mm256_set1_pd(std::numeric_limits<double>::max());
        for (; j + 4 <= i + 1; ) {
            const int c = active[j];
            if (active[j + 3] != c + 3) {
                update_scalar(j++);
                continue;
            }
            __m256d result = _mm256_loadu_pd(target + c);
            for (int p = 0; p < panel.rank; ++p) {
                const std::size_t base = static_cast<std::size_t>(p) * dim;
                const __m256d l = _mm256_set1_pd(panel.left[base + r]);
                const __m256d u = _mm256_loadu_pd(panel.right.data() + base + c);
                result = _mm256_sub_pd(result, _mm256_mul_pd(l, u));
            }
            const __m256d finite = _mm256_cmp_pd(_mm256_andnot_pd(sign, result), limit, _CMP_LE_OQ);
            if (_mm256_movemask_pd(finite) != 15) {
                throw std::runtime_error("Nonfinite blocked LDLT Schur update");
            }
            _mm256_storeu_pd(target + c, result);
            j += 4;
        }
#endif
        for (; j <= i; ++j) { update_scalar(j); }
    }
}

} // anonymous namespace

void factorize_blocked_ldlt(PivotFactor& factor, const GeneralMatrix& input,
                            const Symbolic& symbolic) {
    factor.ready = false;
    if (factor.rank_tolerance <= 0.0 || !std::isfinite(factor.rank_tolerance) ||
        factor.pivot_tolerance <= 0.0 || factor.pivot_tolerance > 0.5 ||
        !std::isfinite(factor.pivot_tolerance) || factor.threads < 1 ||
        factor.block_size < 2 || factor.block_size > 256 ||
        symbolic.parent.size() != static_cast<std::size_t>(input.n)) {
        throw std::runtime_error("Invalid blocked LDLT settings");
    }
#ifndef _OPENMP
    if (factor.threads > 1) {
        throw std::runtime_error("Parallel blocked LDLT requires WITH_OPENMP=1");
    }
#endif
    const int n = input.n;
    factor.matrix = &input;
    factor.blocks.clear();
    factor.rank = factor.positive_directions = factor.negative_directions =
        factor.two_by_two_pivots = 0;
    factor.delayed_candidates = factor.maximum_front = 0;
    factor.blocked_panels = factor.grouped_fronts = 0;
    factor.minimum_pivot = std::numeric_limits<double>::infinity();

    // Validate exactly mirrored symmetric entries and derive lower CSC starts.
    // The same congruent equilibration as the original LDLT path is retained.
    std::vector<std::size_t> lower_start = input.ptr;
    factor.row_scale.assign(n, 0.0);
    for (int c = 0; c < n; ++c) {
        for (auto p = lower_start[c]; p < input.ptr[c + 1]; ++p) {
            const int r = input.row[p];
            const double value = input.value[p];
            if (r < c || !std::isfinite(value)) {
                throw std::runtime_error("LDLT requires canonical symmetric values; use from_symmetric");
            }
            if (r > c) {
                auto& mirror = lower_start[r];
                if (mirror >= input.ptr[r+1] || input.row[mirror] != c ||
                    input.value[mirror] != value) {
                    throw std::runtime_error("LDLT requires canonical symmetric values; use from_symmetric");
                }
                ++mirror;
            }
            const double magnitude = std::abs(value);
            factor.row_scale[r] = std::max(factor.row_scale[r], magnitude);
            factor.row_scale[c] = std::max(factor.row_scale[c], magnitude);
        }
    }
    for (double& s : factor.row_scale) {
        s = s > 0.0 ? 1.0 / std::sqrt(s) : 1.0;
    }
    factor.column_scale = factor.row_scale;

    // Collapse exact unary assembly-tree chains into supernodes without
    // introducing any extra graph fill. The maximum width is a performance
    // policy only: numerical pivot acceptance is still fully dynamic.
    std::vector<int> group_first(n, -1);
    std::vector<int> group_width(n, 0);
    for (int first = 0; first < n; ++first) {
        if (group_first[first] != -1) { continue; }
        int head = first;
        int count = 1;
        group_first[head] = first;
        while (count < factor.block_size) {
            const int parent = symbolic.parent[head];
            if (parent == -1 || symbolic.reuse_child[parent] != head) { break; }
            head = parent;
            group_first[head] = first;
            ++count;
        }
        group_width[head] = count;
    }

    std::vector<BlockFront> contributions(n);
    std::vector<std::vector<PivotBlock>> records(n);
    std::vector<std::size_t> delays(n, 0), widths(n, 0), panels(n, 0), groups(n, 0);
    std::vector<std::vector<int>> worker_positions(factor.threads);
    std::atomic<bool> failed{false};
    std::exception_ptr failure;

    auto process = [&](int owner) {
        if (group_width[owner] == 0) { return; }
        const int bottom = group_first[owner];
        std::vector<int> members;
        members.reserve(group_width[owner]);
        for (int node = bottom;; node = symbolic.parent[node]) {
            members.push_back(node);
            if (node == owner) { break; }
        }
        groups[owner] = 1;

        BlockFront front;
        const auto first = symbolic.row.begin() + symbolic.ptr[bottom];
        const auto last = symbolic.row.begin() + symbolic.ptr[bottom + 1];
        const int reusable = symbolic.reuse_child[bottom];
        const bool inherited = reusable != -1 &&
            contributions[reusable].rows.size() == static_cast<std::size_t>(last - first) &&
            std::equal(first, last, contributions[reusable].rows.begin());
        if (inherited) {
            front = std::move(contributions[reusable]);
            contributions[reusable] = BlockFront{};
        } else {
            front.rows.assign(first, last);
            for (int child = symbolic.first_child[bottom]; child != -1;
                 child = symbolic.next_child[child]) {
                const auto& source = contributions[child];
                front.rows.insert(front.rows.end(), source.rows.begin(), source.rows.end());
            }
            std::sort(front.rows.begin(), front.rows.end());
            front.rows.erase(std::unique(front.rows.begin(), front.rows.end()), front.rows.end());
            const auto width = static_cast<std::size_t>(front.rows.size());
            front.values.assign(width * (width + 1) / 2, 0.0);
            front.stride = static_cast<int>(width);
        }
        const int width = static_cast<int>(front.rows.size());
        widths[owner] = width;
        double* const values = front.values.data();
        auto at = [&](int r, int c) -> double& {
            const auto physical = static_cast<std::size_t>(front.origin + std::max(r,c));
            return values[physical * (physical + 1) / 2 + front.origin + std::min(r,c)];
        };

        // A worker's map is live only while assembling its current front;
        // there is no internal task scheduling in this process.
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        auto& positions = worker_positions[worker];
        if (width >= 16 && positions.empty()) { positions.resize(n); }
        if (width >= 16) {
            for (int i = 0; i < width; ++i) { positions[front.rows[i]] = i; }
        }
        auto locate = [&](int id) {
            return width >= 16 ? positions[id] :
                static_cast<int>(std::lower_bound(front.rows.begin(), front.rows.end(), id) -
                                 front.rows.begin());
        };

        // Assemble every original lower column of the supernode exactly once.
        // Each higher member's column is already in the bottom member's pattern.
        for (int member : members) {
            const int pivot = locate(member);
            for (auto p = lower_start[member]; p < input.ptr[member + 1]; ++p) {
                const int r = input.row[p];
                at(locate(r), pivot) += (input.value[p] * factor.row_scale[r]) *
                                       factor.row_scale[member];
            }
        }

        // Extend-add completed child Schur blocks. A child's delayed variables
        // travel with its rows and can enlarge this symbolic supernode.
        for (int child = symbolic.first_child[bottom]; child != -1;
             child = symbolic.next_child[child]) {
            if (inherited && child == reusable) { continue; }
            auto& source = contributions[child];
            std::vector<int> child_map;
            child_map.reserve(source.rows.size());
            for (int id : source.rows) { child_map.push_back(locate(id)); }
            for (std::size_t r = 0; r < child_map.size(); ++r) {
                const std::size_t sr = static_cast<std::size_t>(source.origin) + r;
                const double* const from = source.values.data() +
                    sr * (sr + 1) / 2 + source.origin;
                const std::size_t tr = static_cast<std::size_t>(front.origin + child_map[r]);
                double* const to = values + tr * (tr + 1) / 2 + front.origin;
                for (std::size_t c = 0; c <= r; ++c) { to[child_map[c]] += from[c]; }
            }
            source = BlockFront{};
        }

        std::vector<int> active(width);
        std::iota(active.begin(), active.end(), 0);
        PivotPanel panel(width, factor.block_size);

        auto current = [&](int r, int c) {
            double result = at(r,c);
            for (int p = 0; p < panel.rank; ++p) {
                const auto base = static_cast<std::size_t>(p) * width;
                result -= panel.left[base+r] * panel.right[base+c];
            }
            if (!std::isfinite(result)) {
                throw std::runtime_error("Nonfinite blocked LDLT candidate");
            }
            return result;
        };
        auto flush = [&] {
            if (panel.rank == 0) { return; }
            rank_k_update(front.origin, width, active, panel, values);
            ++panels[owner];
            panel.clear();
        };

        while (!active.empty()) {
            // Limit a panel's rank; a 2x2 pivot must not straddle its boundary.
            if (panel.rank >= factor.block_size - 1) { flush(); }
            const auto eligible_end = std::upper_bound(
                active.begin(), active.end(), owner,
                [&](int id, int local) { return id < front.rows[local]; });
            const auto eligible_count = static_cast<std::size_t>(eligible_end - active.begin());
            if (eligible_count == 0) { break; }

            int pivots[2] = {-1, -1};
            int count = 1;
            double best = factor.rank_tolerance;
            for (std::size_t i = 0; i < eligible_count; ++i) {
                const int r = active[i];
                const double magnitude = std::abs(current(r,r));
                if (!(magnitude > best)) { continue; }
                double maximum = 0.0;
                for (int c : active) {
                    maximum = std::max(maximum, std::abs(current(r,c)));
                }
                if (magnitude >= factor.pivot_tolerance * maximum) {
                    pivots[0] = r;
                    best = magnitude;
                }
            }

            // A sufficiently stable 2x2 block handles zero-diagonal saddle
            // systems without perturbations or a positive-definiteness guess.
            if (pivots[0] == -1 && eligible_count >= 2) {
                int a = -1, b = -1;
                double strongest = 0.0;
                for (std::size_t i = 0; i < eligible_count; ++i) {
                    for (std::size_t j = 0; j < i; ++j) {
                        const double value = std::abs(current(active[i], active[j]));
                        if (value > strongest) {
                            strongest = value;
                            a = active[j];
                            b = active[i];
                        }
                    }
                }
                if (a != -1 && strongest > factor.rank_tolerance) {
                    const double scale = std::max({
                        std::abs(current(a,a)), strongest, std::abs(current(b,b))});
                    const double d0 = current(a,a) / scale;
                    const double d1 = current(b,b) / scale;
                    const double off = current(a,b) / scale;
                    const double det = d0*d1 - off*off;
                    const double radius = std::hypot(0.5*(d0-d1),off);
                    const double center = 0.5*(d0+d1);
                    const double large = center >= 0.0 ? center+radius : center-radius;
                    const double small = large != 0.0 ? det/large : 0.0;
                    double growth = 0.0;
                    if (std::min(std::abs(large), std::abs(small)) * scale >
                        factor.rank_tolerance) {
                        for (int r : active) {
                            if (r == a || r == b) { continue; }
                            const double x = current(r,a) / scale;
                            const double y = current(r,b) / scale;
                            growth = std::max({
                                growth, std::abs((x*d1-y*off)/det),
                                std::abs((y*d0-x*off)/det)});
                        }
                        if (growth <= 1.0/factor.pivot_tolerance) {
                            count = 2;
                            pivots[0] = a;
                            pivots[1] = b;
                        }
                    }
                }
            }
            if (pivots[0] == -1) { break; }

            PivotBlock block;
            block.size = count;
            for (int p = 0; p < count; ++p) {
                block.rows[p] = block.columns[p] = front.rows[pivots[p]];
                for (int q = 0; q < count; ++q) {
                    block.diagonal[2*p+q] = current(pivots[p],pivots[q]);
                }
            }
            const double scale = count == 1 ? std::abs(block.diagonal[0]) :
                std::max({std::abs(block.diagonal[0]),
                          std::abs(block.diagonal[1]), std::abs(block.diagonal[3])});
            const double a = block.diagonal[0] / scale;
            const double b = block.diagonal[1] / scale;
            const double d = block.diagonal[3] / scale;
            const double det = count == 1 ? a : a*d-b*b;

            // Make room for the new scalar columns. current() still sees only
            // previously accepted pivots until panel.rank is incremented.
            const int base_rank = panel.rank;
            panel.append(count);
            for (int c : active) {
                const auto base = static_cast<std::size_t>(base_rank) * width;
                panel.right[base+c] = current(pivots[0],c);
                if (count == 2) {
                    panel.right[base+width+c] = current(pivots[1],c);
                }
            }
            for (int r : active) {
                if (r == pivots[0] || (count == 2 && r == pivots[1])) { continue; }
                const double x = current(r,pivots[0]) / scale;
                const double y = count == 2 ? current(r,pivots[1]) / scale : 0.0;
                const std::array<double,2> lower = count == 1 ?
                    std::array<double,2>{x/det, 0.0} :
                    std::array<double,2>{(x*d-y*b)/det, (y*a-x*b)/det};
                if (lower[0] != 0.0 || lower[1] != 0.0) {
                    const auto base = static_cast<std::size_t>(base_rank) * width;
                    panel.left[base+r] = lower[0];
                    if (count == 2) { panel.left[base+width+r] = lower[1]; }
                    block.lower_rows.push_back(front.rows[r]);
                    block.lower_values.push_back(lower);
                }
            }
            panel.rank += count;
            active.erase(std::remove_if(active.begin(), active.end(), [&](int r) {
                return r == pivots[0] || (count == 2 && r == pivots[1]);
            }),active.end());
            records[owner].push_back(std::move(block));
        }
        flush();

        if (symbolic.parent[owner] == -1) {
            double maximum = 0.0;
            for (int r : active) {
                for (int c : active) {
                    maximum = std::max(maximum, std::abs(at(r,c)));
                }
            }
            if (maximum > factor.rank_tolerance) {
                throw std::runtime_error(
                    "Unresolved nonzero root block: reduce rank/pivot tolerance; rank is not certified");
            }
        } else {
            auto& output = contributions[owner];
            for (int r : active) {
                output.rows.push_back(front.rows[r]);
                if (front.rows[r] <= owner) { ++delays[owner]; }
            }
            // Pass exact contiguous tails by ownership transfer instead of
            // copying a second packed triangle.
            const int parent = symbolic.parent[owner];
            if (!active.empty() && symbolic.reuse_child[parent] == owner &&
                active.size() == static_cast<std::size_t>(width - active.front()) &&
                front.origin + active.front() <= front.stride/2 &&
                std::equal(output.rows.begin(),output.rows.end(),
                    symbolic.row.begin()+symbolic.ptr[parent],
                    symbolic.row.begin()+symbolic.ptr[parent+1])) {
                output.origin = front.origin + active.front();
                output.stride = front.stride;
                output.values = std::move(front.values);
            } else {
                output.stride = static_cast<int>(active.size());
                output.values.reserve(active.size() * (active.size()+1) / 2);
                for (std::size_t r = 0; r < active.size(); ++r) {
                    for (std::size_t c = 0; c <= r; ++c) {
                        output.values.push_back(at(active[r],active[c]));
                    }
                }
            }
        }
    };

    // Retain independent-subtree task scheduling. Each front owns its factors
    // and contribution buffers, including all private panel workspaces.
    std::function<void(int,int)> visit;
    visit = [&](int root, int depth) {
        if (failed.load(std::memory_order_relaxed)) { return; }
        try {
            if (factor.threads == 1 || symbolic.subtree_size[root] <= 256 ||
                symbolic.subtree_work[root] <= 2e7 || depth >= 64) {
                const int begin = symbolic.subtree_begin[root];
                const int end = begin + symbolic.subtree_size[root];
                for (int p = begin; p < end; ++p) {
                    if (failed.load(std::memory_order_relaxed)) { return; }
                    process(symbolic.postorder[p]);
                }
            } else {
                std::vector<int> chain{root};
                int head = root;
                while (symbolic.first_child[head] != -1 &&
                       symbolic.next_child[symbolic.first_child[head]] == -1) {
                    head = symbolic.first_child[head];
                    chain.push_back(head);
                }
#ifdef _OPENMP
#pragma omp taskgroup
#endif
                {
                    for (int child = symbolic.first_child[head]; child != -1;
                         child = symbolic.next_child[child]) {
#ifdef _OPENMP
#pragma omp task firstprivate(child,depth) shared(visit)
#endif
                        visit(child,depth+1);
                    }
                }
                if (failed.load(std::memory_order_relaxed)) { return; }
                for (auto p = chain.rbegin(); p != chain.rend(); ++p) { process(*p); }
            }
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(mf_blocked_ldlt_failure)
#endif
            { if (!failure) { failure = std::current_exception(); } }
            failed.store(true, std::memory_order_relaxed);
        }
    };
#ifdef _OPENMP
    if (factor.threads > 1) {
#pragma omp parallel num_threads(factor.threads)
        {
#pragma omp single
            {
#pragma omp taskgroup
                {
                    for (int root : symbolic.roots) {
#pragma omp task firstprivate(root) shared(visit)
                        visit(root,0);
                    }
                }
            }
        }
    } else
#endif
    { for (int root : symbolic.roots) { visit(root,0); } }
    if (failure) { std::rethrow_exception(failure); }

    for (int owner : symbolic.postorder) {
        factor.delayed_candidates += delays[owner];
        factor.maximum_front = std::max(factor.maximum_front, widths[owner]);
        factor.blocked_panels += panels[owner];
        factor.grouped_fronts += groups[owner];
        for (auto& block : records[owner]) {
            factor.rank += block.size;
            if (block.size == 1) {
                factor.minimum_pivot = std::min(
                    factor.minimum_pivot,std::abs(block.diagonal[0]));
                if (block.diagonal[0] > 0.0) { ++factor.positive_directions; }
                else { ++factor.negative_directions; }
            } else {
                ++factor.two_by_two_pivots;
                const double scale = std::max({
                    std::abs(block.diagonal[0]),std::abs(block.diagonal[1]),
                    std::abs(block.diagonal[3])});
                const double a = block.diagonal[0]/scale;
                const double b = block.diagonal[1]/scale;
                const double d = block.diagonal[3]/scale;
                const double center = 0.5*(a+d);
                const double radius = std::hypot(0.5*(a-d),b);
                const double large = center >= 0.0 ? center+radius : center-radius;
                const double small = (a*d-b*b)/large;
                if (large > 0.0) { ++factor.positive_directions; }
                else { ++factor.negative_directions; }
                if (small > 0.0) { ++factor.positive_directions; }
                else { ++factor.negative_directions; }
                factor.minimum_pivot = std::min({
                    factor.minimum_pivot,std::abs(large)*scale,std::abs(small)*scale});
            }
            factor.blocks.push_back(std::move(block));
        }
    }
    if (factor.rank == 0) { factor.minimum_pivot = 0.0; }
    factor.ready = true;
}
} // namespace mf
