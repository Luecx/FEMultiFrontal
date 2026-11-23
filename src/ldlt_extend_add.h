/**
 * @file ldlt_extend_add.h
 * @brief Contiguous-run transfer of packed child Schur rows into parent fronts.
 *
 * The source and destination remain in the existing row-packed lower triangle.
 * This is an optional assembly micro-optimization, not a different matrix
 * storage policy. A planned sequence of disjoint child-column runs is reused
 * for every physical child row; arbitrary gaps remain supported.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mf::dense {

struct ExtendAddRun {
    int child_begin = 0;
    int parent_begin = 0;
    int length = 0;
};

struct ExtendAddPlan {
    std::vector<ExtendAddRun> runs;
    bool enabled = false;
};

// The child_map is sorted; map[i] denotes the child's local column i in the
// assembled parent's local index space. The heuristic avoids paying an extra
// run-loop for fragmented child maps where ordinary indexed scatter wins.
inline ExtendAddPlan plan_segmented_extend_add(const std::vector<int>& child_map) {
    ExtendAddPlan plan;
    const int size = static_cast<int>(child_map.size());
    if (size < 32) { return plan; }
    int in_long_runs = 0;
    for (int first = 0; first < size;) {
        int end = first + 1;
        while (end < size && child_map[end] == child_map[end-1] + 1) { ++end; }
        const int length = end-first;
        plan.runs.push_back({first,child_map[first],length});
        if (length >= 8) { in_long_runs += length; }
        first = end;
    }
    plan.enabled =
        in_long_runs >= (size*3)/4 &&
        static_cast<int>(plan.runs.size()) <= std::max(2,size/8);
    if (!plan.enabled) { plan.runs.clear(); }
    return plan;
}

// from[c] and to[child_map[c]] point into packed physical rows. No reordering
// or grouping of contributions across children; every target receives exactly
// the original one addition, so association and floating-point addition order
// are preserved. On diagonal rows only c <= child_row is meaningful.
inline void extend_add_row_segmented(
    double* to, const double* from, int child_row,
    const ExtendAddPlan& plan) {
    for (const auto& run : plan.runs) {
        if (run.child_begin > child_row) { break; }
        const int usable = std::min(run.length, child_row-run.child_begin+1);
        const double* const src = from+run.child_begin;
        double* const dst = to+run.parent_begin;
        int j=0;
#if defined(__AVX2__)
        for (;j+4<=usable;j+=4) {
            const __m256d old = _mm256_loadu_pd(dst+j);
            const __m256d added = _mm256_loadu_pd(src+j);
            _mm256_storeu_pd(dst+j,_mm256_add_pd(old,added));
        }
#endif
        for (;j<usable;++j) { dst[j]+=src[j]; }
    }
}

} // namespace mf::dense
