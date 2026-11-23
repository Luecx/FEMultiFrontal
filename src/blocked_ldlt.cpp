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
#include "ldlt_schur.h"
#include "ldlt_indexed_schur.h"
#include "ldlt_tiled_front.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
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
    bool tiled = false;

    // r/c are logical indices in rows; origin maps them to the inherited
    // physical backing allocation. The tiled variant keeps exactly this
    // coordinate convention so ownership transfers need no reindexing.
    const double& at(int r, int c) const {
        const int physical_row = origin + std::max(r,c);
        const int physical_col = origin + std::min(r,c);
        const std::size_t index = tiled
            ? dense::tiled_front_offset(physical_row,physical_col)
            : static_cast<std::size_t>(physical_row)*(physical_row+1)/2
              + physical_col;
        return values[index];
    }
};

// One writer per front: scalar pivot headers and all multiplier entries are
// appended into shared front-local arrays, with no allocations per pivot.
struct FrontPivotRecords {
    std::vector<PackedSymmetricPivot> blocks;
    std::vector<int> lower_rows;
    std::vector<double> lower_values;
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
            // Use eight AVX2 accumulators for a contiguous 32-column run.
            // The sixteen/eight/four-wide and scalar paths remain unchanged.
            if (j + 32 <= i + 1 && active[j + 31] == c + 31) {
                __m256d result0 = _mm256_loadu_pd(target + c);
                __m256d result1 = _mm256_loadu_pd(target + c + 4);
                __m256d result2 = _mm256_loadu_pd(target + c + 8);
                __m256d result3 = _mm256_loadu_pd(target + c + 12);
                __m256d result4 = _mm256_loadu_pd(target + c + 16);
                __m256d result5 = _mm256_loadu_pd(target + c + 20);
                __m256d result6 = _mm256_loadu_pd(target + c + 24);
                __m256d result7 = _mm256_loadu_pd(target + c + 28);
                for (int p = 0; p < panel.rank; ++p) {
                    const std::size_t base = static_cast<std::size_t>(p) * dim;
                    const __m256d l = _mm256_set1_pd(panel.left[base + r]);
                    const double* const u = panel.right.data() + base + c;
                    result0 = _mm256_sub_pd(result0, _mm256_mul_pd(l, _mm256_loadu_pd(u)));
                    result1 = _mm256_sub_pd(result1, _mm256_mul_pd(l, _mm256_loadu_pd(u + 4)));
                    result2 = _mm256_sub_pd(result2, _mm256_mul_pd(l, _mm256_loadu_pd(u + 8)));
                    result3 = _mm256_sub_pd(result3, _mm256_mul_pd(l, _mm256_loadu_pd(u + 12)));
                    result4 = _mm256_sub_pd(result4, _mm256_mul_pd(l, _mm256_loadu_pd(u + 16)));
                    result5 = _mm256_sub_pd(result5, _mm256_mul_pd(l, _mm256_loadu_pd(u + 20)));
                    result6 = _mm256_sub_pd(result6, _mm256_mul_pd(l, _mm256_loadu_pd(u + 24)));
                    result7 = _mm256_sub_pd(result7, _mm256_mul_pd(l, _mm256_loadu_pd(u + 28)));
                }
                const __m256d finite0 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result0), limit, _CMP_LE_OQ);
                const __m256d finite1 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result1), limit, _CMP_LE_OQ);
                const __m256d finite2 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result2), limit, _CMP_LE_OQ);
                const __m256d finite3 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result3), limit, _CMP_LE_OQ);
                const __m256d finite4 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result4), limit, _CMP_LE_OQ);
                const __m256d finite5 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result5), limit, _CMP_LE_OQ);
                const __m256d finite6 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result6), limit, _CMP_LE_OQ);
                const __m256d finite7 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result7), limit, _CMP_LE_OQ);
                if (_mm256_movemask_pd(finite0) != 15 ||
                    _mm256_movemask_pd(finite1) != 15 ||
                    _mm256_movemask_pd(finite2) != 15 ||
                    _mm256_movemask_pd(finite3) != 15 ||
                    _mm256_movemask_pd(finite4) != 15 ||
                    _mm256_movemask_pd(finite5) != 15 ||
                    _mm256_movemask_pd(finite6) != 15 ||
                    _mm256_movemask_pd(finite7) != 15) {
                    throw std::runtime_error("Nonfinite blocked LDLT Schur update");
                }
                _mm256_storeu_pd(target + c, result0);
                _mm256_storeu_pd(target + c + 4, result1);
                _mm256_storeu_pd(target + c + 8, result2);
                _mm256_storeu_pd(target + c + 12, result3);
                _mm256_storeu_pd(target + c + 16, result4);
                _mm256_storeu_pd(target + c + 20, result5);
                _mm256_storeu_pd(target + c + 24, result6);
                _mm256_storeu_pd(target + c + 28, result7);
                j += 32;
                continue;
            }
            // Consume sixteen contiguous columns with four independent AVX2
            // accumulators. This amortizes the pivot broadcast and rank-loop
            // overhead on large fronts, retaining exact per-entry pivot order.
            if (j + 16 <= i + 1 && active[j + 15] == c + 15) {
                __m256d result0 = _mm256_loadu_pd(target + c);
                __m256d result1 = _mm256_loadu_pd(target + c + 4);
                __m256d result2 = _mm256_loadu_pd(target + c + 8);
                __m256d result3 = _mm256_loadu_pd(target + c + 12);
                for (int p = 0; p < panel.rank; ++p) {
                    const std::size_t base = static_cast<std::size_t>(p) * dim;
                    const __m256d l = _mm256_set1_pd(panel.left[base + r]);
                    const double* const u = panel.right.data() + base + c;
                    result0 = _mm256_sub_pd(result0, _mm256_mul_pd(l, _mm256_loadu_pd(u)));
                    result1 = _mm256_sub_pd(result1, _mm256_mul_pd(l, _mm256_loadu_pd(u + 4)));
                    result2 = _mm256_sub_pd(result2, _mm256_mul_pd(l, _mm256_loadu_pd(u + 8)));
                    result3 = _mm256_sub_pd(result3, _mm256_mul_pd(l, _mm256_loadu_pd(u + 12)));
                }
                const __m256d finite0 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result0), limit, _CMP_LE_OQ);
                const __m256d finite1 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result1), limit, _CMP_LE_OQ);
                const __m256d finite2 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result2), limit, _CMP_LE_OQ);
                const __m256d finite3 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result3), limit, _CMP_LE_OQ);
                if (_mm256_movemask_pd(finite0) != 15 || _mm256_movemask_pd(finite1) != 15 ||
                    _mm256_movemask_pd(finite2) != 15 || _mm256_movemask_pd(finite3) != 15) {
                    throw std::runtime_error("Nonfinite blocked LDLT Schur update");
                }
                _mm256_storeu_pd(target + c, result0);
                _mm256_storeu_pd(target + c + 4, result1);
                _mm256_storeu_pd(target + c + 8, result2);
                _mm256_storeu_pd(target + c + 12, result3);
                j += 16;
                continue;
            }
            // Eight/four-wide fallback preserves the previous kernel for
            // shorter consecutive runs; holes retain scalar handling.
            if (j + 8 <= i + 1 && active[j + 7] == c + 7) {
                __m256d result0 = _mm256_loadu_pd(target + c);
                __m256d result1 = _mm256_loadu_pd(target + c + 4);
                for (int p = 0; p < panel.rank; ++p) {
                    const std::size_t base = static_cast<std::size_t>(p) * dim;
                    const __m256d l = _mm256_set1_pd(panel.left[base + r]);
                    const double* const u = panel.right.data() + base + c;
                    result0 = _mm256_sub_pd(result0, _mm256_mul_pd(l, _mm256_loadu_pd(u)));
                    result1 = _mm256_sub_pd(result1, _mm256_mul_pd(l, _mm256_loadu_pd(u + 4)));
                }
                const __m256d finite0 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result0), limit, _CMP_LE_OQ);
                const __m256d finite1 = _mm256_cmp_pd(_mm256_andnot_pd(sign, result1), limit, _CMP_LE_OQ);
                if (_mm256_movemask_pd(finite0) != 15 || _mm256_movemask_pd(finite1) != 15) {
                    throw std::runtime_error("Nonfinite blocked LDLT Schur update");
                }
                _mm256_storeu_pd(target + c, result0);
                _mm256_storeu_pd(target + c + 4, result1);
                j += 8;
                continue;
            }
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
    const int max_supernode_size = factor.supernode_size == 0
        ? factor.block_size : factor.supernode_size;
    if (max_supernode_size < 1 || max_supernode_size > 256) {
        throw std::runtime_error("Invalid blocked LDLT supernode size");
    }
    factor.matrix = &input;
    factor.blocks.clear();
    factor.packed_blocks.clear();
    factor.packed_lower_rows.clear();
    factor.packed_lower_values.clear();
    factor.rank = factor.positive_directions = factor.negative_directions =
        factor.two_by_two_pivots = 0;
    factor.delayed_candidates = factor.maximum_front = 0;
    factor.blocked_panels = factor.grouped_fronts = 0;
    factor.fast_path_pivots = factor.fallback_searches = factor.relaxed_groups = 0;
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

    // Group exact links and cheaply amalgamate nearly identical unary fronts.
    // Relaxed merges admit only a few additional structural rows; branching
    // nodes are kept separate so all children are fully assembled first.
    // The dense front below always unions the patterns of every member.
    std::vector<int> group_first(n, -1);
    std::vector<int> group_width(n, 0);
    for (int first = 0; first < n; ++first) {
        if (group_first[first] != -1) { continue; }
        int head = first;
        int count = 1;
        group_first[head] = first;
        while (count < max_supernode_size) {
            const int parent = symbolic.parent[head];
            if (parent == -1) { break; }
            const bool exact = symbolic.reuse_child[parent] == head;
            if (!exact) {
                const bool unary = symbolic.first_child[parent] == head &&
                                   symbolic.next_child[head] == -1;
                const int current_size = static_cast<int>(symbolic.ptr[head+1]-symbolic.ptr[head]);
                const int next_size = static_cast<int>(symbolic.ptr[parent+1]-symbolic.ptr[parent]);
                const int extra = next_size - (current_size-1);
                if (!factor.relaxed_supernodes || !unary ||
                    extra > std::max(4,current_size/32)) { break; }
            }
            head = parent;
            group_first[head] = first;
            ++count;
        }
        group_width[head] = count;
    }

    std::vector<BlockFront> contributions(n);
    std::vector<FrontPivotRecords> records(n);
    std::vector<std::size_t> delays(n, 0), widths(n, 0), panels(n, 0), groups(n, 0);
    std::vector<std::size_t> fast_pivots(n, 0), fallback_searches(n, 0), relaxed_groups(n, 0);
    using Clock = std::chrono::steady_clock;
    const bool profile = std::getenv("FEMF_LDLT_PROFILE") != nullptr;
    std::vector<double> assembly_s(n,0.0), search_s(n,0.0), schur_s(n,0.0), front_s(n,0.0);
    // Fine-grained bins are disjoint from each other and from the coarse
    // assembly/search/Schur bins. One writer per front avoids atomics.
    // Keep the fine-grained arrays empty unless profiling is requested.
    std::vector<double> setup_s(profile ? n : 0,0.0);
    std::vector<double> cache_refresh_s(profile ? n : 0,0.0);
    std::vector<double> cache_apply_s(profile ? n : 0,0.0);
    std::vector<double> panel_build_s(profile ? n : 0,0.0);
    std::vector<double> pivot_bookkeeping_s(profile ? n : 0,0.0);
    std::vector<double> contribution_s(profile ? n : 0,0.0);
    // Subphases of assembly and contribution; included in the parent bins,
    // not additional elapsed time.
    std::vector<double> front_prepare_s(profile ? n : 0,0.0);
    std::vector<double> original_assemble_s(profile ? n : 0,0.0);
    std::vector<double> extend_add_s(profile ? n : 0,0.0);
    std::vector<double> contribution_copy_s(profile ? n : 0,0.0);
    std::vector<double> contribution_move_s(profile ? n : 0,0.0);
    std::vector<std::size_t> contribution_copy_entries(profile ? n : 0,0);
    std::vector<std::size_t> contribution_moved_fronts(profile ? n : 0,0);
    // Subphases of front_prepare_s, not additional time.
    std::vector<double> front_row_union_s(profile ? n : 0,0.0);
    std::vector<double> front_zero_s(profile ? n : 0,0.0);
    std::vector<double> front_position_map_s(profile ? n : 0,0.0);
    std::vector<std::size_t> front_contained_sources(profile ? n : 0,0);
    std::vector<std::size_t> front_merged_sources(profile ? n : 0,0);
    std::vector<std::size_t> tiled_fronts(profile ? n : 0,0);
    std::vector<std::size_t> tiled_flushes(profile ? n : 0,0);
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
        const auto started = profile ? Clock::now() : Clock::time_point{};
        bool exact_group = true;
        for (std::size_t k = 1; k < members.size(); ++k) {
            if (symbolic.reuse_child[members[k]] != members[k-1]) {
                exact_group = false;
            }
        }
        relaxed_groups[owner] = !exact_group;
        records[owner].blocks.reserve(members.size());

        BlockFront front;
        const auto first = symbolic.row.begin() + symbolic.ptr[bottom];
        const auto last = symbolic.row.begin() + symbolic.ptr[bottom + 1];
        const int reusable = symbolic.reuse_child[bottom];
        const bool inherited = exact_group && reusable != -1 &&
            contributions[reusable].rows.size() == static_cast<std::size_t>(last - first) &&
            std::equal(first, last, contributions[reusable].rows.begin());
        const auto row_union_started = profile ? Clock::now() : Clock::time_point{};
        if (inherited) {
            front = std::move(contributions[reusable]);
            contributions[reusable] = BlockFront{};
        } else if (factor.fast_front_prepare) {
            // Symbolic column rows and completed child contribution rows are
            // sorted. Build their union incrementally in linear time per input
            // list rather than concatenate duplicates and re-sort the entire
            // potentially large frontal row pattern.
            front.rows.assign(first, last);
            std::vector<int> merged;
            auto add_sorted = [&](auto begin, auto end) {
                if (begin==end) { return; }
                if (std::includes(front.rows.begin(),front.rows.end(),begin,end)) {
                    if (profile) { ++front_contained_sources[owner]; }
                    return;
                }
                merged.clear();
                merged.reserve(front.rows.size()+static_cast<std::size_t>(end-begin));
                std::set_union(front.rows.begin(),front.rows.end(),begin,end,
                               std::back_inserter(merged));
                front.rows.swap(merged);
                if (profile) { ++front_merged_sources[owner]; }
            };
            // The bottom column is already in front.rows: never add it twice.
            for (std::size_t k=1;k<members.size();++k) {
                const int member=members[k];
                add_sorted(symbolic.row.begin()+symbolic.ptr[member],
                           symbolic.row.begin()+symbolic.ptr[member+1]);
            }
            for (int child=symbolic.first_child[bottom]; child!=-1;
                 child=symbolic.next_child[child]) {
                const auto& rows=contributions[child].rows;
                add_sorted(rows.begin(),rows.end());
            }
            // Preserve the original strict-unique row invariant even if an
            // input range contains repeated indices.
            front.rows.erase(std::unique(front.rows.begin(),front.rows.end()),front.rows.end());
        } else {
            // Exact original assembly for benchmark A/B and rollback.
            front.rows.assign(first, last);
            for (int member : members) {
                const auto begin = symbolic.row.begin()+symbolic.ptr[member];
                const auto end = symbolic.row.begin()+symbolic.ptr[member+1];
                front.rows.insert(front.rows.end(),begin,end);
            }
            for (int child = symbolic.first_child[bottom]; child != -1;
                 child = symbolic.next_child[child]) {
                const auto& source = contributions[child];
                front.rows.insert(front.rows.end(), source.rows.begin(), source.rows.end());
            }
            std::sort(front.rows.begin(), front.rows.end());
            front.rows.erase(std::unique(front.rows.begin(), front.rows.end()), front.rows.end());
        }
        if (profile) {
            front_row_union_s[owner] += std::chrono::duration<double>(
                Clock::now()-row_union_started).count();
        }
        if (!inherited) {
            // The frontal matrix must remain fully zero-initialized: original
            // columns and child contributions are subsequently added into it.
            const auto zero_started = profile ? Clock::now() : Clock::time_point{};
            const auto width = static_cast<std::size_t>(front.rows.size());
            // Tile-major storage is enabled only for substantial fronts.
            // Padding all the thousands of tiny multifrontal leaves to 32x32
            // would waste memory without offering any Schur acceleration.
            front.tiled = factor.tiled_front_storage && width >= 96;
            const auto entries = front.tiled
                ? dense::tiled_front_entries(width)
                : width*(width+1)/2;
            front.values.assign(entries, 0.0);
            front.stride = static_cast<int>(width);
            if (profile) {
                front_zero_s[owner] += std::chrono::duration<double>(
                    Clock::now()-zero_started).count();
            }
        }
        const int width = static_cast<int>(front.rows.size());
        widths[owner] = width;
        if (profile && front.tiled) { ++tiled_fronts[owner]; }
        double* const values = front.values.data();
        auto at = [&](int r, int c) -> double& {
            const int physical_row = front.origin + std::max(r,c);
            const int physical_col = front.origin + std::min(r,c);
            if (front.tiled) {
                return values[dense::tiled_front_offset(physical_row,physical_col)];
            }
            return values[static_cast<std::size_t>(physical_row)*(physical_row+1)/2
                          + physical_col];
        };

        // A worker's map is live only while assembling its current front;
        // there is no internal task scheduling in this process.
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        const auto position_map_started = profile ? Clock::now() : Clock::time_point{};
        auto& positions = worker_positions[worker];
        if (width >= 16 && positions.empty()) { positions.resize(n); }
        if (width >= 16) {
            for (int i = 0; i < width; ++i) { positions[front.rows[i]] = i; }
        }
        if (profile) {
            front_position_map_s[owner] += std::chrono::duration<double>(
                Clock::now()-position_map_started).count();
        }
        auto locate = [&](int id) {
            return width >= 16 ? positions[id] :
                static_cast<int>(std::lower_bound(front.rows.begin(), front.rows.end(), id) -
                                 front.rows.begin());
        };

        if (profile) {
            front_prepare_s[owner] += std::chrono::duration<double>(
                Clock::now()-started).count();
        }
        const auto original_started = profile ? Clock::now() : Clock::time_point{};
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

        if (profile) {
            original_assemble_s[owner] += std::chrono::duration<double>(
                Clock::now()-original_started).count();
        }
        const auto extend_started = profile ? Clock::now() : Clock::time_point{};
        // Extend-add completed child Schur blocks. A child's delayed variables
        // travel with its rows and can enlarge this symbolic supernode.
        for (int child = symbolic.first_child[bottom]; child != -1;
             child = symbolic.next_child[child]) {
            if (inherited && child == reusable) { continue; }
            auto& source = contributions[child];
            std::vector<int> child_map;
            child_map.reserve(source.rows.size());
            for (int id : source.rows) { child_map.push_back(locate(id)); }
            if (!front.tiled && !source.tiled) {
                // Preserve the incumbent fast streaming extend-add on the
                // standard row-packed layout.
                for (std::size_t r = 0; r < child_map.size(); ++r) {
                    const std::size_t sr = static_cast<std::size_t>(source.origin) + r;
                    const double* const from = source.values.data() +
                        sr * (sr + 1) / 2 + source.origin;
                    const std::size_t tr = static_cast<std::size_t>(front.origin + child_map[r]);
                    double* const to = values + tr * (tr + 1) / 2 + front.origin;
                    for (std::size_t c = 0; c <= r; ++c) { to[child_map[c]] += from[c]; }
                }
            } else {
                // Mixed layouts are valid: a tiled front may inherit a small
                // row-packed child and vice versa. Never reinterpret the
                // child's physical backing storage as the parent's layout.
                for (std::size_t r = 0; r < child_map.size(); ++r) {
                    for (std::size_t c = 0; c <= r; ++c) {
                        at(child_map[r],child_map[c]) +=
                            source.at(static_cast<int>(r),static_cast<int>(c));
                    }
                }
            }
            source = BlockFront{};
        }

        if (profile) {
            extend_add_s[owner] += std::chrono::duration<double>(
                Clock::now()-extend_started).count();
            assembly_s[owner] += std::chrono::duration<double>(Clock::now()-started).count();
        }
        const auto setup_started = profile ? Clock::now() : Clock::time_point{};
        std::vector<int> active(width);
        std::iota(active.begin(), active.end(), 0);
        PivotPanel panel(width, factor.block_size);

        // Incrementally maintain fully summed candidate diagonals. Keep up
        // to eight complete candidate rows as a tiny pivot workspace so the
        // threshold test does not re-evaluate rank-length dot products for
        // the same row on every accepted pivot.
        constexpr int cache_capacity = 8;
        std::vector<double> diagonal;
        std::vector<double> cached_values;
        std::vector<int> cache_slot;
        std::array<int,cache_capacity> cached_ids{};
        cached_ids.fill(-1);
        if (factor.cached_pivot_rows) {
            diagonal.resize(width);
            cached_values.resize(static_cast<std::size_t>(cache_capacity)*width);
            cache_slot.assign(width,-1);
            for (int r=0;r<width;++r) { diagonal[r]=at(r,r); }
        }
        if (profile) {
            setup_s[owner] += std::chrono::duration<double>(Clock::now()-setup_started).count();
        }
        auto raw_current = [&](int r,int c) {
            double result=at(r,c);
            for (int p=0;p<panel.rank;++p) {
                const auto base=static_cast<std::size_t>(p)*width;
                result-=panel.left[base+r]*panel.right[base+c];
            }
            if (!std::isfinite(result)) {
                throw std::runtime_error("Nonfinite blocked LDLT candidate");
            }
            return result;
        };
        auto current = [&](int r,int c) {
            if (factor.cached_pivot_rows) {
                if (r==c) { return diagonal[r]; }
                if (cache_slot[r]>=0) {
                    return cached_values[static_cast<std::size_t>(cache_slot[r])*width+c];
                }
                if (cache_slot[c]>=0) {
                    return cached_values[static_cast<std::size_t>(cache_slot[c])*width+r];
                }
            }
            return raw_current(r,c);
        };
        auto refresh_candidates = [&](std::size_t count) {
            if (!factor.cached_pivot_rows) { return; }
            const int keep=static_cast<int>(std::min<std::size_t>(cache_capacity,count));
            // Only cache rows in the current lookahead. Delayed rows can
            // remain unchanged for many iterations, retaining their slots.
            for (int slot=0;slot<cache_capacity;++slot) {
                const int r=cached_ids[slot];
                if (r<0) { continue; }
                bool needed=false;
                for (int j=0;j<keep;++j) {
                    if (active[j]==r) { needed=true; break; }
                }
                if (!needed) {
                    cache_slot[r]=-1;
                    cached_ids[slot]=-1;
                }
            }
            for (int j=0;j<keep;++j) {
                const int r=active[j];
                if (cache_slot[r]>=0) { continue; }
                int slot=0;
                while (slot<cache_capacity && cached_ids[slot]>=0) { ++slot; }
                if (slot==cache_capacity) {
                    throw std::runtime_error("LDLT candidate cache exhausted");
                }
                // This is done exactly once when a candidate enters the
                // lookahead, not once per failed threshold check.
                double* row=cached_values.data()+static_cast<std::size_t>(slot)*width;
                for (int c:active) { row[c]=raw_current(r,c); }
                cached_ids[slot]=r;
                cache_slot[r]=slot;
            }
        };
        auto apply_new_pivots_to_cache = [&](int base_rank,int count) {
            if (!factor.cached_pivot_rows) { return; }
            for (int p=base_rank;p<base_rank+count;++p) {
                const auto base=static_cast<std::size_t>(p)*width;
                for (int r:active) {
                    diagonal[r]-=panel.left[base+r]*panel.right[base+r];
                    if (!std::isfinite(diagonal[r])) {
                        throw std::runtime_error("Nonfinite LDLT cached diagonal");
                    }
                }
                for (int slot=0;slot<cache_capacity;++slot) {
                    const int r=cached_ids[slot];
                    if (r<0) { continue; }
                    const double multiplier=panel.left[base+r];
                    double* row=cached_values.data()+static_cast<std::size_t>(slot)*width;
                    for (int c:active) {
                        row[c]-=multiplier*panel.right[base+c];
                        if (!std::isfinite(row[c])) {
                            throw std::runtime_error("Nonfinite LDLT cached candidate row");
                        }
                    }
                }
            }
        };
        auto flush = [&] {
            if (panel.rank == 0) { return; }
            const auto t = profile ? Clock::now() : Clock::time_point{};
            if (front.tiled) {
                if (profile) { ++tiled_flushes[owner]; }
                dense::rank_k_ldlt_tiled_front(
                    front.origin, width, active, panel.rank,
                    panel.left.data(), panel.right.data(), values);
            } else if (factor.tiled_schur && active.size() >= 96 && panel.rank >= 8) {
                if (factor.indexed_schur) {
                    dense::rank_k_ldlt_indexed_packed(
                        front.origin, width, active, panel.rank,
                        panel.left.data(), panel.right.data(), values);
                } else {
                    dense::rank_k_ldlt_tiled(
                        front.origin, width, active, panel.rank,
                        panel.left.data(), panel.right.data(), values);
                }
            } else {
                rank_k_update(front.origin, width, active, panel, values);
            }
            if (profile) {
                schur_s[owner] += std::chrono::duration<double>(Clock::now()-t).count();
            }
            ++panels[owner];
            panel.clear();
            if (factor.cached_pivot_rows) {
                for (int slot=0;slot<cache_capacity;++slot) { cached_ids[slot]=-1; }
                std::fill(cache_slot.begin(),cache_slot.end(),-1);
                // Cached diagonals already include every accepted correction.
                // Fresh rows will be rebuilt from the flushed physical front.
            }
        };

        while (!active.empty()) {
            // Limit a panel's rank; a 2x2 pivot must not straddle its boundary.
            if (panel.rank >= factor.block_size - 1) { flush(); }
            const auto eligible_end = std::upper_bound(
                active.begin(), active.end(), owner,
                [&](int id, int local) { return id < front.rows[local]; });
            const auto eligible_count = static_cast<std::size_t>(eligible_end - active.begin());
            if (eligible_count == 0) { break; }
            const auto cache_refresh_started = profile ? Clock::now() : Clock::time_point{};
            refresh_candidates(eligible_count);
            if (profile) {
                cache_refresh_s[owner] +=
                    std::chrono::duration<double>(Clock::now()-cache_refresh_started).count();
            }

            const auto search_started = profile ? Clock::now() : Clock::time_point{};
            int pivots[2] = {-1, -1};
            int count = 1;

            // Most structural FE pivots need no global max-diagonal search.
            // Test the first fully summed candidate against the ENTIRE active
            // row (including delayed update variables), not only the panel.
            // If it fails, preserve the original exhaustive threshold search
            // and 2x2 fallback without changing either acceptance criterion.
            if (factor.fast_pivoting) {
                // A delayed near-zero candidate may remain at the beginning
                // of active for many steps. Inspect a short lookahead instead
                // of re-running the exhaustive search every such iteration.
                const std::size_t lookahead = std::min<std::size_t>(eligible_count,8);
                for (std::size_t i = 0; i < lookahead; ++i) {
                    const int r = active[i];
                    const double magnitude = std::abs(current(r,r));
                    if (!(magnitude > factor.rank_tolerance)) { continue; }
                    double maximum = magnitude;
                    for (int c : active) {
                        maximum = std::max(maximum,std::abs(current(r,c)));
                        if (magnitude < factor.pivot_tolerance*maximum) { break; }
                    }
                    if (magnitude >= factor.pivot_tolerance*maximum) {
                        pivots[0] = r;
                        ++fast_pivots[owner];
                        break;
                    }
                }
            }
            if (pivots[0] == -1) {
                ++fallback_searches[owner];
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
            if (profile) {
                search_s[owner] += std::chrono::duration<double>(Clock::now()-search_started).count();
            }
            if (pivots[0] == -1) { break; }

            // Includes forming the pivot block, computing factor multipliers,
            // appending the L/R panel and writing per-pivot multiplier records.
            const auto panel_build_started = profile ? Clock::now() : Clock::time_point{};
            PackedSymmetricPivot block;
            block.size = count;
            auto& owner_records = records[owner];
            block.lower_begin = owner_records.lower_rows.size();
            block.value_begin = owner_records.lower_values.size();
            for (int p = 0; p < count; ++p) {
                block.rows[p] = front.rows[pivots[p]];
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
                // Symmetric pivot-to-row couplings were already evaluated
                // above for the panel's right factor. Reuse them instead of
                // recomputing current() and its rank-dependent dot product.
                const double x = panel.right[base_rank * static_cast<std::size_t>(width) + r] / scale;
                const double y = count == 2
                    ? panel.right[(base_rank + 1) * static_cast<std::size_t>(width) + r] / scale
                    : 0.0;
                const std::array<double,2> lower = count == 1 ?
                    std::array<double,2>{x/det, 0.0} :
                    std::array<double,2>{(x*d-y*b)/det, (y*a-x*b)/det};
                if (lower[0] != 0.0 || lower[1] != 0.0) {
                    const auto base = static_cast<std::size_t>(base_rank) * width;
                    panel.left[base+r] = lower[0];
                    if (count == 2) { panel.left[base+width+r] = lower[1]; }
                    owner_records.lower_rows.push_back(front.rows[r]);
                    owner_records.lower_values.push_back(lower[0]);
                    if (count == 2) { owner_records.lower_values.push_back(lower[1]); }
                }
            }
            panel.rank += count;
            if (profile) {
                panel_build_s[owner] +=
                    std::chrono::duration<double>(Clock::now()-panel_build_started).count();
            }

            const auto bookkeeping_started = profile ? Clock::now() : Clock::time_point{};
            active.erase(std::remove_if(active.begin(), active.end(), [&](int r) {
                return r == pivots[0] || (count == 2 && r == pivots[1]);
            }),active.end());
            if (factor.cached_pivot_rows) {
                for (int p=0;p<count;++p) {
                    const int slot=cache_slot[pivots[p]];
                    if (slot>=0) {
                        cached_ids[slot]=-1;
                        cache_slot[pivots[p]]=-1;
                    }
                }
            }
            if (profile) {
                pivot_bookkeeping_s[owner] +=
                    std::chrono::duration<double>(Clock::now()-bookkeeping_started).count();
            }

            const auto cache_apply_started = profile ? Clock::now() : Clock::time_point{};
            apply_new_pivots_to_cache(base_rank,count);
            if (profile) {
                cache_apply_s[owner] +=
                    std::chrono::duration<double>(Clock::now()-cache_apply_started).count();
            }

            const auto record_started = profile ? Clock::now() : Clock::time_point{};
            block.lower_count = owner_records.lower_rows.size() - block.lower_begin;
            owner_records.blocks.push_back(block);
            if (profile) {
                pivot_bookkeeping_s[owner] +=
                    std::chrono::duration<double>(Clock::now()-record_started).count();
            }
        }
        flush();

        // Root check and inter-front Schur contribution packaging/inheritance.
        // Includes allocations, active-index mapping and triangular copying.
        const auto contribution_started = profile ? Clock::now() : Clock::time_point{};
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
                const auto move_started = profile ? Clock::now() : Clock::time_point{};
                output.origin = front.origin + active.front();
                output.stride = front.stride;
                output.values = std::move(front.values);
                output.tiled = front.tiled;
                if (profile) {
                    contribution_move_s[owner] += std::chrono::duration<double>(
                        Clock::now()-move_started).count();
                    ++contribution_moved_fronts[owner];
                }
            } else {
                const auto copy_started = profile ? Clock::now() : Clock::time_point{};
                output.stride = static_cast<int>(active.size());
                output.values.reserve(active.size() * (active.size()+1) / 2);
                if (front.tiled) {
                    // A parent that cannot inherit this tile-major physical
                    // allocation still expects a canonical compact triangle.
                    // Convert contiguous active suffixes one tile-row segment
                    // at a time instead of doing an indexed lookup per entry.
                    const std::size_t nactive=active.size();
                    output.values.resize(nactive*(nactive+1)/2);
                    const int first_active=active.front();
                    const bool contiguous=static_cast<std::size_t>(
                        active.back()-first_active+1)==nactive;
                    for (std::size_t r=0; r<nactive; ++r) {
                        double* target=output.values.data()+r*(r+1)/2;
                        const int physical_row=front.origin+active[r];
                        if (contiguous) {
                            int physical_col=front.origin+first_active;
                            const int end_col=physical_row+1;
                            while (physical_col<end_col) {
                                const int remaining_in_tile =
                                    dense::front_tile_size - physical_col%dense::front_tile_size;
                                const int count=std::min(end_col-physical_col,remaining_in_tile);
                                const double* source=values+
                                    dense::tiled_front_offset(physical_row,physical_col);
                                std::copy_n(source,count,target);
                                target+=count;
                                physical_col+=count;
                            }
                        } else {
                            // Noncontiguous delayed pivots leave holes.
                            for (std::size_t c=0;c<=r;++c) {
                                target[c]=at(active[r],active[c]);
                            }
                        }
                    }
                } else if (factor.fast_contribution_copy && !active.empty()) {
                    // Active indices are sorted. For each output row, the
                    // source lies entirely below the physical diagonal.
                    // Avoid repeated max/min in at() for every copied entry.
                    const int first_active = active.front();
                    const bool contiguous =
                        static_cast<std::size_t>(active.back()-first_active+1) == active.size();
                    for (std::size_t r=0; r<active.size(); ++r) {
                        const std::size_t physical = static_cast<std::size_t>(
                            front.origin+active[r]);
                        const double* row = values +
                            physical*(physical+1)/2 + front.origin;
                        if (contiguous) {
                            // One bulk copy per packed output row; input may
                            // have a nonzero inherited front origin.
                            output.values.insert(
                                output.values.end(), row+first_active,
                                row+first_active+r+1);
                        } else {
                            // Delayed-pivot holes: copy only the active entries
                            // in canonical packed order; no dense-front repack.
                            for (std::size_t c=0; c<=r; ++c) {
                                output.values.push_back(row[active[c]]);
                            }
                        }
                    }
                } else {
                    // Exact incumbent path for A/B measurements and rollback.
                    for (std::size_t r=0; r<active.size(); ++r) {
                        for (std::size_t c=0; c<=r; ++c) {
                            output.values.push_back(at(active[r],active[c]));
                        }
                    }
                }
                if (profile) {
                    contribution_copy_s[owner] += std::chrono::duration<double>(
                        Clock::now()-copy_started).count();
                    contribution_copy_entries[owner] += active.size()*(active.size()+1)/2;
                }
            }
        }
        if (profile) {
            contribution_s[owner] +=
                std::chrono::duration<double>(Clock::now()-contribution_started).count();
            front_s[owner] = std::chrono::duration<double>(Clock::now()-started).count();
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

    const auto finalize_started = profile ? Clock::now() : Clock::time_point{};
    // Build a single, contiguous factor representation in elimination order.
    // Reserve once after all parallel fronts have finished to avoid reallocations
    // and preserve the original symbolic postorder across independent subtrees.
    std::size_t block_total = 0, row_total = 0, value_total = 0;
    for (const auto& record : records) {
        block_total += record.blocks.size();
        row_total += record.lower_rows.size();
        value_total += record.lower_values.size();
    }
    factor.packed_blocks.reserve(block_total);
    factor.packed_lower_rows.reserve(row_total);
    factor.packed_lower_values.reserve(value_total);
    for (int owner : symbolic.postorder) {
        factor.fast_path_pivots += fast_pivots[owner];
        factor.fallback_searches += fallback_searches[owner];
        factor.relaxed_groups += relaxed_groups[owner];
        factor.delayed_candidates += delays[owner];
        factor.maximum_front = std::max(factor.maximum_front, widths[owner]);
        factor.blocked_panels += panels[owner];
        factor.grouped_fronts += groups[owner];
        const auto& record = records[owner];
        const std::size_t row_offset = factor.packed_lower_rows.size();
        const std::size_t value_offset = factor.packed_lower_values.size();
        factor.packed_lower_rows.insert(factor.packed_lower_rows.end(),
                                        record.lower_rows.begin(), record.lower_rows.end());
        factor.packed_lower_values.insert(factor.packed_lower_values.end(),
                                          record.lower_values.begin(), record.lower_values.end());
        for (auto block : record.blocks) {
            block.lower_begin += row_offset;
            block.value_begin += value_offset;
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
            factor.packed_blocks.push_back(block);
        }
    }
    if (factor.rank == 0) { factor.minimum_pivot = 0.0; }
    if (profile) {
        const double factor_pack_s =
            std::chrono::duration<double>(Clock::now()-finalize_started).count();
        auto sum = [](const std::vector<double>& xs) {
            return std::accumulate(xs.begin(),xs.end(),0.0);
        };
        // A residual bin is computed only from disjoint measurement intervals.
        // It contains front traversal, uninstrumented work and timer overhead,
        // not additional factorization time. This is CPU-time-like work when
        // fronts execute concurrently; individual phase sums may overlap.
        const double measured = sum(assembly_s)+sum(search_s)+sum(schur_s)+
            sum(setup_s)+sum(cache_refresh_s)+sum(cache_apply_s)+
            sum(panel_build_s)+sum(pivot_bookkeeping_s)+sum(contribution_s);
        const double unattributed = sum(front_s)-measured;
        const auto sum_count = [](const std::vector<std::size_t>& xs) {
            return std::accumulate(xs.begin(),xs.end(),std::size_t{0});
        };
        std::fprintf(stderr,
            "FEMF_LDLT_PROFILE fronts=%zu relaxed=%zu fast_pivots=%zu fallback_searches=%zu "
            "assembly_s=%.6f pivot_search_s=%.6f schur_s=%.6f total_front_s=%.6f "
            "front_setup_s=%.6f cache_refresh_s=%.6f cache_apply_s=%.6f "
            "panel_build_s=%.6f pivot_bookkeeping_s=%.6f contribution_s=%.6f "
            "unattributed_front_s=%.6f factor_pack_s=%.6f "
            "front_prepare_s=%.6f original_assemble_s=%.6f extend_add_s=%.6f "
            "contribution_copy_s=%.6f contribution_move_s=%.6f "
            "contribution_copy_entries=%zu contribution_moved_fronts=%zu "
            "front_row_union_s=%.6f front_zero_s=%.6f front_position_map_s=%.6f "
            "front_contained_sources=%zu front_merged_sources=%zu "
            "tiled_fronts=%zu tiled_flushes=%zu "
            "(phase sums overlap with parallel work; subphases included in parent bins)\n",
            factor.grouped_fronts,factor.relaxed_groups,factor.fast_path_pivots,
            factor.fallback_searches,sum(assembly_s),sum(search_s),
            sum(schur_s),sum(front_s),sum(setup_s),sum(cache_refresh_s),
            sum(cache_apply_s),sum(panel_build_s),sum(pivot_bookkeeping_s),
            sum(contribution_s),unattributed,factor_pack_s,
            sum(front_prepare_s),sum(original_assemble_s),sum(extend_add_s),
            sum(contribution_copy_s),sum(contribution_move_s),
            sum_count(contribution_copy_entries),sum_count(contribution_moved_fronts),
            sum(front_row_union_s),sum(front_zero_s),sum(front_position_map_s),
            sum_count(front_contained_sources),sum_count(front_merged_sources),
            sum_count(tiled_fronts),sum_count(tiled_flushes));
    }
    factor.ready = true;
}
} // namespace mf
