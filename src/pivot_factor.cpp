/**
 * @file pivot_factor.cpp
 * @brief Implements dynamic multifrontal threshold-pivoted LDLT and LU.
 *
 * Original entries belong to the front at min(row,column). Child Schur blocks
 * carry both update variables and delayed fully summed rows/columns. A pivot
 * may use only IDs whose original front has arrived; unresolved candidates move
 * upward. This preserves multifrontal ownership under independent LU row/column
 * choices and symmetric two-variable pivots.
 */
#include "pivot_factor.h"
#include "dense_pivot.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf {

/**
 * @brief Owned dynamic Schur block passed from a numerical front to its parent.
 *
 * Row/column sets may differ after LU pivoting, even though this square system
 * retains equal counts. Symmetric fronts store the packed lower triangle.
 * LU fronts store independent rows and columns in a full row-major buffer.
 * An exact symmetric suffix may keep its original stride and diagonal origin
 * across a chain of fronts. The logical index sets describe only live entries.
 * A panel of accepted scalar pivots may accompany that same buffer. Its
 * coefficient columns use the physical stride, and travel only on exact links.
 * Every buffer has one consuming parent; unrelated subtrees own disjoint blocks.
 */
struct NumericFront {
    std::vector<int> rows;
    std::vector<int> columns;
    memory::Values values;
    memory::Values pending_lower;
    memory::Values pending_upper;
    int stride = 0;
    int origin = 0;
};

/**
 * Assemble dynamic fronts and accept stable fully summed numerical pivots.
 *
 * Two-sided square-root maximum scaling bounds initial entries by one without
 * forming normal equations. Symmetric mode requires exactly symmetric stored
 * values, as supplied by GeneralMatrix::from_symmetric. A structural union
 * tree supplies assembly dependencies, while dynamic child sets carry delays.
 * Accepted blocks update the complete trailing front; no static SPD L pattern
 * is assumed. A nonzero unresolved root is an error, not an invented nullmode.
 */
void PivotFactor::factorize(const GeneralMatrix& input, const Symbolic& symbolic) {
    // Validate mode, tolerances and the structural ownership contract
    ready = false;
    if (rank_tolerance <= 0.0 || !std::isfinite(rank_tolerance) || pivot_tolerance <= 0.0 ||
        pivot_tolerance > 0.5 || !std::isfinite(pivot_tolerance) || threads < 1 ||
        symbolic.parent.size() != static_cast<std::size_t>(input.n)) {
        throw std::runtime_error("Invalid pivot settings or assembly-tree dimension");
    }
#ifndef _OPENMP
    if (threads > 1) { throw std::runtime_error("Pivoted parallel factorization requires WITH_OPENMP=1"); }
#endif
    const bool symmetric = kind == Kind::Symmetric;
    const int n = input.n;
    matrix = &input;
    blocks.clear();
    rank = positive_directions = negative_directions = two_by_two_pivots = 0;
    delayed_candidates = maximum_front = 0;
    minimum_pivot = std::numeric_limits<double>::infinity();
    row_scale.assign(n, 0.0);
    column_scale.assign(n, 0.0);

    std::vector<std::size_t> lower_start;
    std::vector<std::size_t> row_ptr;
    std::vector<int> row_columns;
    memory::Values row_values;
    if (symmetric) {
        // Match transposed entries with monotone cursors. As lower columns
        // arrive in order, their counterparts are the next upper entries.
        // Retain the lower start per column for direct assembly.
        lower_start = input.ptr;
        for (int c = 0; c < n; ++c) {
            for (auto p = lower_start[c]; p < input.ptr[c+1]; ++p) {
                const int r = input.row[p];
                const double value = input.value[p];
                if (r < c || (r == c && std::isnan(value))) {
                    throw std::runtime_error("LDLT requires canonical symmetric values; use from_symmetric");
                }
                if (r > c) {
                    auto& mirror = lower_start[r];
                    if (mirror >= input.ptr[r+1] || input.row[mirror] != c || input.value[mirror] != value) {
                        throw std::runtime_error("LDLT requires canonical symmetric values; use from_symmetric");
                    }
                    ++mirror;
                }
                const double magnitude = std::abs(value);
                row_scale[r] = std::max(row_scale[r], magnitude);
                row_scale[c] = std::max(row_scale[c], magnitude);
            }
        }
        column_scale = row_scale;
    } else {
        // LU requires independent row access and separate scaling in each space
        row_ptr.assign(static_cast<std::size_t>(n)+1, 0);
        for (int c = 0; c < n; ++c) {
            for (auto p = input.ptr[c]; p < input.ptr[c+1]; ++p) {
                const int r = input.row[p];
                ++row_ptr[r+1];
                row_scale[r] = std::max(row_scale[r], std::abs(input.value[p]));
                column_scale[c] = std::max(column_scale[c], std::abs(input.value[p]));
            }
        }
        std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        row_columns.resize(input.row.size());
        row_values.resize(input.value.size());
        auto next = row_ptr;
        for (int c = 0; c < n; ++c) {
            for (auto p = input.ptr[c]; p < input.ptr[c+1]; ++p) {
                const auto q = next[input.row[p]]++;
                row_columns[q] = c;
                row_values[q] = input.value[p];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        row_scale[i]    = row_scale[i] > 0.0 ? 1.0 / std::sqrt(row_scale[i]) : 1.0;
        column_scale[i] = column_scale[i] > 0.0 ? 1.0 / std::sqrt(column_scale[i]) : 1.0;
    }

    // Dynamic payloads and accepted records have one writer per original front
    std::vector<NumericFront> contributions(n);
    std::vector<std::vector<PivotBlock>> records(n);
    std::vector<std::size_t> delays(n, 0);
    std::vector<std::size_t> widths(n, 0);
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::vector<std::vector<int>> worker_positions(symmetric ? threads : 0);
    auto process = [&](int owner) {
        // An exact unary match can consume a child buffer directly. Delayed
        // rows can change the index set, so check the actual outgoing IDs.
        NumericFront front;
        const auto first = symbolic.row.begin() + symbolic.ptr[owner];
        const auto last  = symbolic.row.begin() + symbolic.ptr[owner+1];
        const int child_to_reuse = symmetric && symbolic.reuse_child.size() == static_cast<std::size_t>(n)
            ? symbolic.reuse_child[owner] : -1;
        const bool inherited = child_to_reuse != -1 &&
            std::equal(first, last, contributions[child_to_reuse].rows.begin(), contributions[child_to_reuse].rows.end());
        if (inherited) {
            front = std::move(contributions[child_to_reuse]);
            contributions[child_to_reuse] = NumericFront{};
        } else {
            // Union symbolic future indices with all delayed child IDs
            front.rows.assign(first, last);
            front.columns = front.rows;
            for (int child = symbolic.first_child[owner]; child != -1; child = symbolic.next_child[child]) {
                front.rows.insert(front.rows.end(), contributions[child].rows.begin(), contributions[child].rows.end());
                front.columns.insert(front.columns.end(), contributions[child].columns.begin(), contributions[child].columns.end());
            }
            std::sort(front.rows.begin(), front.rows.end());
            std::sort(front.columns.begin(), front.columns.end());
            front.rows.erase(std::unique(front.rows.begin(), front.rows.end()), front.rows.end());
            front.columns.erase(std::unique(front.columns.begin(), front.columns.end()), front.columns.end());
        }
        if (symmetric && front.rows != front.columns) { throw std::runtime_error("LDLT frontal index invariant failed"); }
        const int nr = static_cast<int>(front.rows.size());
        const int nc = static_cast<int>(front.columns.size());
        widths[owner] = std::max(nr, nc);
        if (!inherited) {
            front.stride = nc;
            const auto rows = static_cast<std::size_t>(nr);
            front.values.assign(symmetric ? rows*(rows+1)/2 : rows*nc, 0.0);
        }
        double* const values = front.values.data();
        auto row_position = [&](int id) {
            return static_cast<int>(std::lower_bound(front.rows.begin(), front.rows.end(), id) - front.rows.begin());
        };
        auto column_position = [&](int id) {
            return static_cast<int>(std::lower_bound(front.columns.begin(), front.columns.end(), id) - front.columns.begin());
        };
        auto at = [&](int r, int c) -> double& {
            if (symmetric) {
                const auto row = static_cast<std::size_t>(front.origin + std::max(r,c));
                return values[row*(row+1)/2 + front.origin + std::min(r,c)];
            }
            return values[static_cast<std::size_t>(r)*front.stride+c];
        };

        // Only the owner is eligible when a pending panel arrives. Bring its
        // row and column up to date before adding the owner's original entries.
        if (!front.pending_lower.empty()) {
            for (std::size_t base = 0; base < front.pending_lower.size(); base += front.stride) {
                const double upper = front.pending_upper[base+front.origin];
                for (int r = 0; r < nr; ++r) {
                    const double lower = front.pending_lower[base+front.origin+r];
#if defined(__FMA__)
                    at(r,0) -= std::fma(lower, upper, 0.0);
#else
                    at(r,0) -= lower*upper+0.0;
#endif
                    if (!std::isfinite(at(r,0))) { throw std::runtime_error("Nonfinite pivoted Schur panel column"); }
                }
            }
        }
        // The owner column is already current. Flush only the untouched suffix.
        auto flush_pending = [&] {
            if (front.pending_lower.empty()) { return; }
            dense::flush_scalar_pivot_updates(
                front.stride, front.origin+1, front.origin+nc,
                static_cast<int>(front.pending_lower.size()/front.stride),
                front.pending_lower.data(), front.pending_upper.data(), front.values.data()
            );
            front.pending_lower.clear();
            front.pending_upper.clear();
        };

        // The worker map is used only during assembly, before any taskloop
        // can suspend this front and run another front on the same worker.
        std::vector<int>* positions = nullptr;
        if (symmetric && nr >= 16) {
            int worker = 0;
#ifdef _OPENMP
            worker = omp_get_thread_num();
#endif
            auto& map = worker_positions[worker];
            if (map.empty()) { map.resize(n); }
            for (int i = 0; i < nr; ++i) { map[front.rows[i]] = i; }
            positions = &map;
        }
        auto assembly_row = [&](int id) { return positions ? (*positions)[id] : row_position(id); };
        auto assembly_column = [&](int id) { return positions ? (*positions)[id] : column_position(id); };

        // Assemble the stored lower triangle directly from lower CSC. Scale
        // each value in the same order as the original lower-half assembly.
        if (symmetric) {
            const int pivot = assembly_row(owner);
            for (auto p = lower_start[owner]; p < input.ptr[owner+1]; ++p) {
                const int r = input.row[p];
                const int local = assembly_row(r);
                const double value = input.value[p];
                at(local,pivot) += (value*row_scale[r])*column_scale[owner];
            }
        } else {
            // Insert every original full entry once at min(row,column), in scaled coordinates
            for (std::size_t p = input.ptr[owner]; p < input.ptr[owner + 1]; ++p) {
                const int r = input.row[p];
                if (r >= owner) { at(assembly_row(r), assembly_column(owner)) += (input.value[p] * row_scale[r]) * column_scale[owner]; }
            }
            for (std::size_t p = row_ptr[owner]; p < row_ptr[owner + 1]; ++p) {
                const int c = row_columns[p];
                if (c > owner) { at(assembly_row(owner), assembly_column(c)) += (row_values[p] * row_scale[owner]) * column_scale[c]; }
            }
        }

        // Extend-add child residuals, including previously postponed candidate variables
        for (int child = symbolic.first_child[owner]; child != -1; child = symbolic.next_child[child]) {
            if (inherited && child == child_to_reuse) { continue; }
            auto& source = contributions[child];
            std::vector<int> rows;
            std::vector<int> columns;
            for (int id : source.rows) { rows.push_back(assembly_row(id)); }
            if (!symmetric) {
                for (int id : source.columns) { columns.push_back(assembly_column(id)); }
            }
            const auto& local_columns = symmetric ? rows : columns;
            for (std::size_t r = 0; r < rows.size(); ++r) {
                const auto source_row = r + source.origin;
                const auto base = symmetric ? source_row*(source_row+1)/2 : source_row*source.stride;
                const auto count = symmetric ? r+1 : local_columns.size();
                for (std::size_t c = 0; c < count; ++c) {
                    at(rows[r], local_columns[c]) += source.values[base + source.origin + c];
                }
            }
            source = NumericFront{};
        }
        std::vector<int> active_rows(nr);
        std::vector<int> active_columns(nc);
        std::iota(active_rows.begin(), active_rows.end(), 0);
        std::iota(active_columns.begin(), active_columns.end(), 0);

        // Reuse local positions and contiguous pivot-row coefficients only for
        // sufficiently large symmetric fronts. LU keeps its independent path.
        const bool dense_update = symmetric && nc >= 64;
        std::vector<int> lower_positions;
        std::vector<double> update_upper;
        if (dense_update) {
            lower_positions.reserve(nr);
            update_upper.resize(2*static_cast<std::size_t>(nc));
        }

        // Only original IDs that have reached this front are fully summed candidates
        while (!active_rows.empty() && !active_columns.empty()) {
            // Active IDs stay sorted after pivot removal. Eligible variables
            // occupy a prefix, so pivot search needs no temporary index lists.
            const auto eligible_end = std::upper_bound(active_rows.begin(), active_rows.end(), owner,
                [&](int id, int local) { return id < front.rows[local]; });
            const std::span<const int> eligible_rows(active_rows.begin(), eligible_end);
            auto eligible_columns = eligible_rows;
            if (!symmetric) {
                const auto end = std::upper_bound(active_columns.begin(), active_columns.end(), owner,
                    [&](int id, int local) { return id < front.columns[local]; });
                eligible_columns = std::span<const int>(active_columns.begin(), end);
            }
            if (eligible_rows.empty() || eligible_columns.empty()) { break; }
            int pivot_rows[2] = {-1, -1};
            int pivot_columns[2] = {-1, -1};
            int size = 1;
            double best = rank_tolerance;

            if (symmetric) {
                // Only diagonal candidates can become scalar LDLT pivots.
                // Check the coupling maxima only for a diagonal that can win.
                for (int r : eligible_rows) {
                    const double magnitude = std::abs(at(r,r));
                    if (!(magnitude > best)) { continue; }
                    double maximum = 0.0;
                    for (int c : active_columns) { maximum = std::max(maximum, std::abs(at(r,c))); }
                    if (magnitude >= pivot_tolerance*maximum) {
                        best = magnitude;
                        pivot_rows[0] = pivot_columns[0] = r;
                    }
                }
            } else {
                // General pivots compete independently in row and column space
                std::vector<double> row_max(nr, 0.0);
                std::vector<double> column_max(nc, 0.0);
                for (int r : eligible_rows) {
                    for (int c : active_columns) { row_max[r] = std::max(row_max[r], std::abs(at(r,c))); }
                }
                for (int c : eligible_columns) {
                    for (int r : active_rows) { column_max[c] = std::max(column_max[c], std::abs(at(r,c))); }
                }
                for (int r : eligible_rows) {
                    for (int c : eligible_columns) {
                        const double value = std::abs(at(r,c));
                        if (value > best && value >= pivot_tolerance*row_max[r] && value >= pivot_tolerance*column_max[c]) {
                            best = value;
                            pivot_rows[0] = r;
                            pivot_columns[0] = c;
                        }
                    }
                }
            }

            // A symmetric off-diagonal block can be stable even with zero diagonals
            if (symmetric && pivot_rows[0] == -1 && eligible_rows.size() >= 2) {
                int a = -1;
                int b = -1;
                double strongest = 0.0;
                for (std::size_t i = 0; i < eligible_rows.size(); ++i) {
                    for (std::size_t j = 0; j < i; ++j) {
                        const double value = std::abs(at(eligible_rows[i], eligible_rows[j]));
                        if (value > strongest) { strongest = value; a = eligible_rows[j]; b = eligible_rows[i]; }
                    }
                }
                if (a != -1 && strongest > rank_tolerance) {
                    const double scale = std::max({std::abs(at(a, a)), strongest, std::abs(at(b, b))});
                    const double d0 = at(a, a) / scale;
                    const double d1 = at(b, b) / scale;
                    const double off = at(a, b) / scale;
                    const double det = d0 * d1 - off * off;
                    const double radius = std::hypot(0.5 * (d0 - d1), off);
                    const double center = 0.5 * (d0 + d1);
                    const double large = center >= 0.0 ? center + radius : center - radius;
                    const double small = large != 0.0 ? det / large : 0.0;
                    double growth = 0.0;
                    if (std::min(std::abs(large), std::abs(small)) * scale > rank_tolerance) {
                        for (int r : active_rows) {
                            if (r == a || r == b) { continue; }
                            const double x = at(r, a) / scale;
                            const double y = at(r, b) / scale;
                            growth = std::max({growth, std::abs((x * d1 - y * off) / det),
                                              std::abs((y * d0 - x * off) / det)});
                        }
                        if (growth <= 1.0 / pivot_tolerance) {
                            size = 2;
                            pivot_rows[0] = pivot_columns[0] = a;
                            pivot_rows[1] = pivot_columns[1] = b;
                        }
                    }
                }
            }
            if (pivot_rows[0] == -1) { break; }

            // Store the accepted pivot and its normalized lower multipliers
            lower_positions.clear();
            PivotBlock block;
            block.size = size;
            for (int p = 0; p < size; ++p) {
                block.rows[p] = front.rows[pivot_rows[p]];
                block.columns[p] = front.columns[pivot_columns[p]];
                for (int q = 0; q < size; ++q) { block.diagonal[p * 2 + q] = at(pivot_rows[p], pivot_columns[q]); }
            }
            const double scale = size == 1 ? std::abs(block.diagonal[0]) :
                std::max({std::abs(block.diagonal[0]), std::abs(block.diagonal[1]), std::abs(block.diagonal[3])});
            const double a = block.diagonal[0] / scale;
            const double b = block.diagonal[1] / scale;
            const double d = block.diagonal[3] / scale;
            const double det = size == 1 ? a : a * d - b * b;
            for (int r : active_rows) {
                if (r == pivot_rows[0] || (size == 2 && r == pivot_rows[1])) { continue; }
                const double x = at(r, pivot_columns[0]) / scale;
                const double y = size == 2 ? at(r, pivot_columns[1]) / scale : 0.0;
                const std::array<double, 2> lower = size == 1 ? std::array<double, 2>{x / det, 0.0} :
                    std::array<double, 2>{(x * d - y * b) / det, (y * a - x * b) / det};
                if (lower[0] != 0.0 || lower[1] != 0.0) {
                    if (dense_update) { lower_positions.push_back(r); }
                    block.lower_rows.push_back(front.rows[r]);
                    block.lower_values.push_back(lower);
                }
            }
            std::vector<std::array<double, 2>> upper(dense_update ? 0 : nc);
            for (int c : active_columns) {
                if (dense_update) {
                    update_upper[c] = at(pivot_rows[0], c);
                    if (size == 2) { update_upper[nc+c] = at(pivot_rows[1], c); }
                } else {
                    upper[c] = {at(pivot_rows[0], c), size == 2 ? at(pivot_rows[1], c) : 0.0};
                }
                if (!symmetric && c != pivot_columns[0] && (upper[c][0] != 0.0 || upper[c][1] != 0.0)) {
                    block.upper_columns.push_back(front.columns[c]);
                    block.upper_values.push_back(upper[c]);
                }
            }
            active_rows.erase(std::remove_if(active_rows.begin(), active_rows.end(), [&](int r) {
                return r == pivot_rows[0] || (size == 2 && r == pivot_rows[1]);
            }), active_rows.end());
            active_columns.erase(std::remove_if(active_columns.begin(), active_columns.end(), [&](int c) {
                return c == pivot_columns[0] || (size == 2 && c == pivot_columns[1]);
            }), active_columns.end());

            // Defer only a dense scalar correction whose exact suffix can be
            // handed to the next owner. Zero multipliers use the immediate
            // path to preserve the original rule for mirroring skipped rows.
            constexpr int panel_limit = 16;
            const int parent = symbolic.parent[owner];
            const bool defer = dense_update && size == 1 && pivot_rows[0] == 0 &&
                lower_positions.size() == static_cast<std::size_t>(nc-1) &&
                active_rows.size() == lower_positions.size() &&
                front.rows[0] == owner && front.origin+1 <= front.stride/2 &&
                parent != -1 && symbolic.reuse_child.size() == static_cast<std::size_t>(n) &&
                symbolic.reuse_child[parent] == owner &&
                std::equal(front.rows.begin()+1, front.rows.end(),
                           symbolic.row.begin() + symbolic.ptr[parent], symbolic.row.begin() + symbolic.ptr[parent+1]);
            if (defer) {
                const auto base = front.pending_lower.size();
                front.pending_lower.resize(base+front.stride);
                front.pending_upper.resize(base+front.stride);
                for (int r = 1; r < nc; ++r) {
                    const auto index = base+front.origin+r;
                    front.pending_lower[index] = block.lower_values[r-1][0];
                    front.pending_upper[index] = update_upper[r];
                }
                if (front.pending_lower.size()/front.stride == panel_limit) { flush_pending(); }
            } else {
                flush_pending();
            }

            if (!defer && dense_update) {
                dense::symmetric_pivot_update(
                    front.origin,
                    size,
                    static_cast<int>(lower_positions.size()),
                    lower_positions.data(),
                    block.lower_values.data(),
                    static_cast<int>(active_columns.size()),
                    active_columns.data(),
                    update_upper.data(),
                    update_upper.data()+nc,
                    values
                );
            } else if (!defer) {
                // Update trailing coordinates. Symmetric pairs are evaluated once and mirrored
                for (std::size_t p = 0; p < block.lower_rows.size(); ++p) {
                    const int r = row_position(block.lower_rows[p]);
                    const auto lower = block.lower_values[p];
                    for (int c : active_columns) {
                        if (symmetric && c > r) { continue; }
                        at(r, c) -= lower[0] * upper[c][0] + lower[1] * upper[c][1];
                        if (!std::isfinite(at(r, c))) { throw std::runtime_error("Nonfinite pivoted Schur update"); }
                    }
                }
            }
            records[owner].push_back(std::move(block));
        }

        // Delay unresolved candidates, or deflate only a numerically zero root block
        const bool root = symbolic.parent[owner] == -1;
        if (root) {
            flush_pending();
            double maximum = 0.0;
            for (int r : active_rows) {
                for (int c : active_columns) { maximum = std::max(maximum, std::abs(at(r, c))); }
            }
            if (maximum > rank_tolerance) {
                throw std::runtime_error("Unresolved nonzero root block: reduce rank/pivot tolerance; rank is not certified");
            }
        } else {
            auto& output = contributions[owner];
            for (int r : active_rows) {
                output.rows.push_back(front.rows[r]);
                if (front.rows[r] <= owner) { ++delays[owner]; }
            }
            for (int c : active_columns) { output.columns.push_back(front.columns[c]); }
            // Move contiguous suffixes on exact links. Compact when half the
            // physical rows have been eliminated to bound retained storage.
            const int parent = symbolic.parent[owner];
            if (symmetric && symbolic.reuse_child.size() == static_cast<std::size_t>(n) &&
                symbolic.reuse_child[parent] == owner && !active_rows.empty() &&
                active_rows.size() == static_cast<std::size_t>(nr-active_rows.front()) &&
                active_rows.front() + front.origin <= front.stride/2 &&
                std::equal(output.rows.begin(), output.rows.end(),
                           symbolic.row.begin() + symbolic.ptr[parent], symbolic.row.begin() + symbolic.ptr[parent+1])) {
                output.stride = front.stride;
                output.origin = front.origin + active_rows.front();
                // A rejected pivot leaves the materialized owner in the front.
                // Its delay changes the outgoing IDs, so pending panels only
                // travel on the accepted-pivot branch with a positive offset.
                if (active_rows.front() == 0) { flush_pending(); }
                output.values        = std::move(front.values);
                output.pending_lower = std::move(front.pending_lower);
                output.pending_upper = std::move(front.pending_upper);
            } else {
                flush_pending();
                output.stride = static_cast<int>(active_columns.size());
                const auto rows = active_rows.size();
                output.values.reserve(symmetric ? rows*(rows+1)/2 : rows*active_columns.size());
                for (std::size_t r = 0; r < rows; ++r) {
                    const auto count = symmetric ? r+1 : active_columns.size();
                    for (std::size_t c = 0; c < count; ++c) {
                        output.values.push_back(at(active_rows[r],active_columns[c]));
                    }
                }
            }
        }
    };

    // Schedule coarse independent subtrees; pivot selection remains local and deterministic
    std::function<void(int, int)> visit;
    visit = [&](int root, int depth) {
        if (failed.load(std::memory_order_relaxed)) { return; }
        try {
            if (threads == 1 || symbolic.subtree_size[root] <= 256 || symbolic.subtree_work[root] <= 2e7 || depth >= 64) {
                const int begin = symbolic.subtree_begin[root];
                const int end = begin + symbolic.subtree_size[root];
                for (int p = begin; p < end; ++p) {
                    if (failed.load(std::memory_order_relaxed)) { return; }
                    process(symbolic.postorder[p]);
                }
            } else {
                std::vector<int> chain{root};
                int head = root;
                while (symbolic.first_child[head] != -1 && symbolic.next_child[symbolic.first_child[head]] == -1) {
                    head = symbolic.first_child[head]; chain.push_back(head);
                }
#ifdef _OPENMP
#pragma omp taskgroup
#endif
                {
                    for (int child = symbolic.first_child[head]; child != -1; child = symbolic.next_child[child]) {
#ifdef _OPENMP
#pragma omp task firstprivate(child, depth) shared(visit)
#endif
                        visit(child, depth + 1);
                    }
                }
                if (failed.load(std::memory_order_relaxed)) { return; }
                for (auto p = chain.rbegin(); p != chain.rend(); ++p) { process(*p); }
            }
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(mf_pivot_failure)
#endif
            { if (!failure) { failure = std::current_exception(); } }
            failed.store(true, std::memory_order_relaxed);
        }
    };
#ifdef _OPENMP
    if (threads > 1) {
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            {
#pragma omp taskgroup
                { for (int root : symbolic.roots) {
#pragma omp task firstprivate(root) shared(visit)
                    visit(root, 0);
                } }
            }
        }
    } else
#endif
    { for (int root : symbolic.roots) { visit(root, 0); } }
    if (failure) { std::rethrow_exception(failure); }

    // Flatten accepted records in dependency order and compute tolerance-policy diagnostics
    for (int owner : symbolic.postorder) {
        delayed_candidates += delays[owner];
        maximum_front = std::max(maximum_front, widths[owner]);
        for (auto& block : records[owner]) {
            rank += block.size;
            if (block.size == 1) {
                minimum_pivot = std::min(minimum_pivot, std::abs(block.diagonal[0]));
                if (symmetric) { block.diagonal[0] > 0.0 ? ++positive_directions : ++negative_directions; }
            } else {
                ++two_by_two_pivots;
                const double scale = std::max({std::abs(block.diagonal[0]), std::abs(block.diagonal[1]), std::abs(block.diagonal[3])});
                const double a = block.diagonal[0] / scale;
                const double b = block.diagonal[1] / scale;
                const double d = block.diagonal[3] / scale;
                const double center = 0.5 * (a + d);
                const double radius = std::hypot(0.5 * (a - d), b);
                const double large = center >= 0.0 ? center + radius : center - radius;
                const double determinant = a * d - b * b;
                const double small = determinant / large;
                large > 0.0 ? ++positive_directions : ++negative_directions;
                small > 0.0 ? ++positive_directions : ++negative_directions;
                minimum_pivot = std::min({minimum_pivot, std::abs(large) * scale, std::abs(small) * scale});
            }
            blocks.push_back(std::move(block));
        }
    }
    if (rank == 0) { minimum_pivot = 0.0; }
    ready = true;
}

/**
 * Solve the numerical block factors and check the original unscaled equations.
 *
 * Unpivoted scaled variables are set to zero. A rank-deficient consistent
 * system therefore returns one basic solution. Failure of the original-system
 * residual check reports incompatible data rather than silently solving a
 * regularized or least-squares problem.
 */
std::vector<double> PivotFactor::solve(const std::vector<double>& rhs) const {
    // Apply equation scaling before unit-lower forward substitution
    if (!ready || !matrix || rhs.size() != static_cast<std::size_t>(matrix->n)) {
        throw std::runtime_error("Pivot factors are not ready or RHS dimension is invalid");
    }
    auto y = rhs;
    for (int i = 0; i < matrix->n; ++i) {
        if (!std::isfinite(rhs[i])) { throw std::runtime_error("Nonfinite RHS"); }
        y[i] *= row_scale[i];
    }
    for (const auto& block : blocks) {
        const double a = y[block.rows[0]];
        const double b = block.size == 2 ? y[block.rows[1]] : 0.0;
        for (std::size_t p = 0; p < block.lower_rows.size(); ++p) {
            y[block.lower_rows[p]] -= block.lower_values[p][0] * a + block.lower_values[p][1] * b;
        }
    }

    // Back substitution uses D*L^T for symmetric factors and independent U for LU
    std::vector<double> x(matrix->n, 0.0);
    for (auto item = blocks.rbegin(); item != blocks.rend(); ++item) {
        const auto& block = *item;
        double a = y[block.rows[0]];
        double b = block.size == 2 ? y[block.rows[1]] : 0.0;
        if (kind == Kind::General) {
            for (std::size_t p = 0; p < block.upper_columns.size(); ++p) {
                a -= block.upper_values[p][0] * x[block.upper_columns[p]];
                b -= block.upper_values[p][1] * x[block.upper_columns[p]];
            }
        }
        if (block.size == 1) {
            a /= block.diagonal[0];
        } else {
            const double scale = std::max({std::abs(block.diagonal[0]), std::abs(block.diagonal[1]), std::abs(block.diagonal[3])});
            const double d0 = block.diagonal[0] / scale;
            const double d1 = block.diagonal[3] / scale;
            const double off = block.diagonal[1] / scale;
            const double det = d0 * d1 - off * off;
            const double first = a / scale;
            const double second = b / scale;
            a = (d1 * first - off * second) / det;
            b = (d0 * second - off * first) / det;
        }
        if (kind == Kind::Symmetric) {
            for (std::size_t p = 0; p < block.lower_rows.size(); ++p) {
                a -= block.lower_values[p][0] * x[block.lower_rows[p]];
                b -= block.lower_values[p][1] * x[block.lower_rows[p]];
            }
        }
        x[block.columns[0]] = a;
        if (block.size == 2) { x[block.columns[1]] = b; }
    }
    for (int i = 0; i < matrix->n; ++i) { x[i] *= column_scale[i]; }

    // Verify against A itself, retaining the distinction between singular and inconsistent
    const auto product = matrix->multiply(x);
    long double error = 0.0L;
    long double norm = 0.0L;
    for (int i = 0; i < matrix->n; ++i) {
        const long double difference = static_cast<long double>(product[i]) - rhs[i];
        error += difference * difference;
        norm += static_cast<long double>(rhs[i]) * rhs[i];
    }
    const long double residual = norm > 0.0L ? std::sqrt(error / norm) : std::sqrt(error);
    if (!std::isfinite(residual) || residual > std::max(1e-8, 100.0 * rank_tolerance)) {
        throw std::runtime_error(rank < matrix->n ? "Singular system has an incompatible RHS at the declared tolerance" :
                                                    "Pivoted solution failed the original-system residual check");
    }
    return x;
}

} // namespace mf
