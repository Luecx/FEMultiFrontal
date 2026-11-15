/**
 * @file ordering.cpp
 * @brief Implements multilevel nested dissection on the compressed sparsity graph.
 *
 * The ordering consists of three stages:
 *
 * - Graph compression merges unknowns with identical, or for consecutive
 *   unknowns nearly identical, closed adjacency into supervariables. Their
 *   vertex weight is the number of merged unknowns, so every separator and
 *   degree measure remains a scalar DOF count. Edges are weighted by the number
 *   of common neighbors to guide coarsening along the mesh structure.
 * - Nested dissection recursively computes small balanced vertex separators.
 *   Each separator is found on a hierarchy of heavy-edge-matched coarse graphs,
 *   initialized by graph growing and refined by one-sided node FM moves while
 *   the partition is projected back to the finer graphs. Large subgraphs compare
 *   several separators on an intermediate coarse graph, sharing the expensive
 *   fine-graph hierarchy instead of rebuilding it for independent trials.
 * - Subdomains below a fixed size are ordered by minimum degree. The degree
 *   includes the halo of not yet eliminated separator vertices, which are the
 *   only external neighbors of a subdomain and remain uneliminated.
 *
 * Subdomains are numbered before their separators, so the resulting
 * elimination tree follows the dissection tree. Disconnected components are
 * ordered independently. The ordering inspects the pattern only; numerical
 * values and caller-supplied grouping information are never used.
 *
 * @see Matrix
 * @see Symbolic
 */
#include "ordering.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf {

namespace {

// Minimum closed-neighborhood overlap of consecutive unknowns merged into one group
constexpr double kSimilarity = 0.9;

// Subdomains with at most this many supervariables are ordered by minimum degree
constexpr int kLeafSize = 120;

// Coarsening stops at this graph size or once matching no longer reduces the graph
constexpr int    kCoarsenTo        = 100;
constexpr double kCoarsenReduction = 0.9;

// Grown initial separators on the coarsest graph. Graphs from kIntermediateAbove
// vertices are bisected kIntermediateRuns times on an intermediate coarse graph.
// One fine-graph hierarchy is built per subdomain.
constexpr int kInitialTries      = 8;
constexpr int kIntermediateAbove = 5000;
constexpr int kIntermediateRuns  = 2;

// FM passes per direction, tolerated moves without improvement and the
// maximal weight fraction of either part
constexpr int    kRefinePasses = 10;
constexpr int    kStallMoves   = 400;
constexpr double kMaxPart      = 0.6;

// Subgraphs below this size are dissected inside the task of their parent
constexpr int kTaskSize = 2000;

// Mix 64-bit values into well-distributed hashes and pseudo-random numbers
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/**
 * @brief Undirected weighted graph used by the nested-dissection ordering.
 *
 * Adjacency is stored without self loops in compressed rows. A vertex weight
 * counts the scalar unknowns represented by a supervariable or coarse vertex.
 * An edge weight counts the fine graph edges merged into a coarse edge and
 * steers heavy-edge matching; separators are always measured in vertex weight.
 */
struct Graph {
    // Compressed adjacency rows and the multiplicity of every stored edge
    std::vector<int> ptr{0};
    std::vector<int> adj;
    std::vector<int> edge_weight;

    // Number of scalar unknowns per vertex
    std::vector<int> vertex_weight;

    int size() const { return static_cast<int>(vertex_weight.size()); }
};

/**
 * @brief Addressable binary max-heap of integer separator-move gains.
 *
 * Every vertex owns one key and one heap position. Gains change by vertex
 * weights whenever a neighboring vertex enters or leaves the separator, so the
 * heap supports incremental key changes and removal of arbitrary members.
 * clear() resets only the positions of remaining members.
 */
struct GainHeap {
    // Heap-ordered members, their keys and their heap positions (-1 if absent)
    std::vector<int> items;
    std::vector<int> key;
    std::vector<int> position;

    // Construction and membership
    void resize(int n) {
        items.clear();
        key.assign(n, 0);
        position.assign(n, -1);
    }
    void clear() {
        for (int v : items) { position[v] = -1; }
        items.clear();
    }
    bool empty() const { return items.empty(); }
    int  top() const { return items.front(); }
    bool contains(int v) const { return position[v] != -1; }

    // Insert, adjust and remove members while preserving the heap order
    void insert(int v, int k) {
        key[v]      = k;
        position[v] = static_cast<int>(items.size());
        items.push_back(v);
        sift_up(position[v]);
    }
    void change(int v, int delta) {
        key[v] += delta;
        if (delta > 0) { sift_up(position[v]); } else { sift_down(position[v]); }
    }
    void remove(int v) {
        const int p    = position[v];
        const int last = items.back();
        position[v] = -1;
        items.pop_back();
        if (p < static_cast<int>(items.size())) {
            items[p]       = last;
            position[last] = p;
            sift_up(p);
            sift_down(position[last]);
        }
    }

    // Restore the heap order above or below one position
    void sift_up(int p) {
        const int v = items[p];
        while (p > 0) {
            const int parent = (p - 1) / 2;
            if (key[items[parent]] >= key[v]) { break; }
            items[p]           = items[parent];
            position[items[p]] = p;
            p                  = parent;
        }
        items[p]    = v;
        position[v] = p;
    }
    void sift_down(int p) {
        const int v = items[p];
        const int n = static_cast<int>(items.size());
        while (true) {
            int child = 2 * p + 1;
            if (child >= n) { break; }
            if (child + 1 < n && key[items[child + 1]] > key[items[child]]) { ++child; }
            if (key[items[child]] <= key[v]) { break; }
            items[p]           = items[child];
            position[items[p]] = p;
            p                  = child;
        }
        items[p]    = v;
        position[v] = p;
    }
};

/**
 * @brief Recursive multilevel nested dissection of one compressed graph.
 *
 * The complete compressed graph stays immutable. Every recursive call owns a
 * local subgraph, the global identifiers of its vertices and a contiguous
 * range of output positions. Part zero is placed first, part one second and
 * the separator last, so independent calls write disjoint ranges and can run
 * as concurrent tasks.
 *
 * Leaf ordering reads the complete graph to find the halo of a subdomain. Its
 * global-to-local marker is owned per thread because leaf ordering contains no
 * task scheduling point.
 */
struct NestedDissection {
    // Complete compressed graph and its elimination order by output position
    const Graph&     graph;
    std::vector<int> order;

    // Per-thread global-to-local markers for leaf ordering
    std::vector<std::vector<int>> leaf_local;

    // Coarsening has no task scheduling points. Each worker can reuse its
    // blocks, matching, coarse-member and adjacency-slot arrays across calls.
    mutable std::vector<std::array<std::vector<int>, 4>> coarsen_workspace;

    // Threads available for independent subdomain tasks
    int threads = 1;

    // Recursive dissection and its separator computation. Part labels are 0
    // and 1 for the two parts and 2 for separator vertices.
    void                     dissect(const Graph& g, const std::vector<int>& ids, int first);
    std::vector<signed char> separate(const Graph& g, std::uint64_t seed) const;
    std::vector<signed char> initial_separator(const Graph& g, std::uint64_t& rng, int max_part) const;
    Graph                    coarsen(const Graph& g, std::vector<int>& map, int max_vertex, std::uint64_t& rng) const;
    void                     refine(const Graph& g, std::vector<signed char>& where, int max_part) const;

    // Coarsen, separate the coarsest graph through the callback and refine the
    // projected partition on every finer level
    std::vector<signed char> multilevel(
        const Graph&                                                 g,
        std::uint64_t&                                               rng,
        int                                                          max_part,
        int                                                          coarsen_to,
        int                                                          max_levels,
        const std::function<std::vector<signed char>(const Graph&)>& coarsest
    ) const;

    // Split a labeled graph into the induced subgraphs of labels 0..count-1
    void split(const Graph&                   g,
               const std::vector<int>&        ids,
               const std::vector<int>&        label,
               int                            count,
               std::vector<Graph>&            parts,
               std::vector<std::vector<int>>& part_ids) const;

    // Halo minimum degree of a small subdomain
    void order_leaf(const std::vector<int>& ids, int first);
};

/**
 * Order one subgraph into the output range [first, first + g.size()).
 *
 * Small subgraphs are ordered by halo minimum degree. Larger disconnected
 * subgraphs are split into their components without a separator. A connected
 * subgraph receives a multilevel vertex separator; both parts are dissected
 * recursively, possibly as concurrent tasks, and the separator is numbered
 * last in its local order.
 *
 * The top separators dominate fill and front sizes. Several initial partitions
 * and intermediate-graph trials are compared within separate(), while one
 * fine-graph hierarchy is built per subdomain. Independent child subdomains
 * run concurrently when threads are available.
 *
 * Task synchronization uses task groups. A plain taskwait would also wait
 * for sibling subtrees spawned earlier by the same task and serialize them.
 *
 * A separator that leaves no vertex in either part cannot make progress. This
 * happens only for nearly complete graphs, where every elimination order
 * produces the same dense front, so the vertices then keep their local order.
 *
 * @param g Local subgraph to order.
 * @param ids Global compressed identifiers of the local vertices.
 * @param first First output position owned by this subgraph.
 */
void NestedDissection::dissect(const Graph& g, const std::vector<int>& ids, int first) {
    const int n = g.size();

    // Order small subdomains directly with their separator halo
    if (n <= kLeafSize) {
        order_leaf(ids, first);
        return;
    }

    // Label connected components by breadth-first search
    std::vector<int> label(n, -1);
    std::vector<int> queue;
    queue.reserve(n);
    int components = 0;
    for (int seed = 0; seed < n; ++seed) {
        if (label[seed] != -1) { continue; }
        queue.clear();
        queue.push_back(seed);
        label[seed] = components;
        for (std::size_t k = 0; k < queue.size(); ++k) {
            const int v = queue[k];
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                const int u = g.adj[e];
                if (label[u] == -1) {
                    label[u] = components;
                    queue.push_back(u);
                }
            }
        }
        ++components;
    }

    // Disconnected components need no separator and occupy consecutive ranges
    if (components > 1) {
        std::vector<Graph>            parts;
        std::vector<std::vector<int>> part_ids;
        split(g, ids, label, components, parts, part_ids);
        for (int c = 0; c < components; ++c) {
            dissect(parts[c], part_ids[c], first);
            first += parts[c].size();
        }
        return;
    }

    // Search one hierarchy per subdomain. Repeated coarse-level initializations
    // inside separate() retain separator diversity without rebuilding the large
    // fine-graph hierarchy for several complete independent trials.
    const std::uint64_t seed = mix(static_cast<std::uint64_t>(first) * 0x100000001b3ULL + n);
    std::vector<signed char> where = separate(g, mix(seed));

    // Collect the separator and count both parts
    std::vector<int>   separator;
    std::array<int, 2> count{};
    for (int v = 0; v < n; ++v) {
        label[v] = where[v] == 2 ? -1 : where[v];
        if (where[v] == 2) { separator.push_back(ids[v]); } else { ++count[where[v]]; }
    }

    // Keep the local order when the separator does not split the graph
    if (count[0] == n || count[1] == n || static_cast<int>(separator.size()) == n) {
        std::copy(ids.begin(), ids.end(), order.begin() + first);
        return;
    }

    // Place both parts before their separator and dissect them recursively
    std::vector<Graph>            parts;
    std::vector<std::vector<int>> part_ids;
    split(g, ids, label, 2, parts, part_ids);
    const int left  = parts[0].size();
    const int right = parts[1].size();
    std::copy(separator.begin(), separator.end(), order.begin() + first + left + right);
    label = {};
    where = {};
#ifdef _OPENMP
    if (threads > 1 && std::min(left, right) >= kTaskSize) {
#pragma omp taskgroup
        {
#pragma omp task default(shared) firstprivate(first)
            dissect(parts[0], part_ids[0], first);
            dissect(parts[1], part_ids[1], first + left);
        }
        return;
    }
#endif
    dissect(parts[0], part_ids[0], first);
    dissect(parts[1], part_ids[1], first + left);
}

/**
 * Compute a balanced vertex separator of a connected graph.
 *
 * A complete bisection coarsens the graph by heavy-edge matching to at most
 * 100 vertices, takes the best of several grown and refined separators of the
 * coarsest graph and refines the projected partition on every finer level.
 *
 * Graphs from kIntermediateAbove vertices are first coarsened by at most four
 * levels to an intermediate graph of about 1/30 of their size. This
 * intermediate graph is bisected completely several times, and only the
 * lightest balanced result is refined back to the input graph. The repeated
 * bisections thereby cost a fraction of the fine-level work.
 *
 * @param g Connected graph with at least two vertices.
 * @param seed Seed of the pseudo-random matching and initial seeds.
 * @return Part label per vertex: 0 and 1 for the parts, 2 for the separator.
 */
std::vector<signed char> NestedDissection::separate(const Graph& g, std::uint64_t seed) const {
    // Derive the balance limit from the total weight; coarse graphs keep it
    const int     total    = std::accumulate(g.vertex_weight.begin(), g.vertex_weight.end(), 0);
    const int     max_part = std::max(static_cast<int>(kMaxPart * total), (total + 1) / 2);
    std::uint64_t rng      = seed;

    // Complete bisection: coarsen to a small graph and grow initial separators there
    auto bisect = [&](const Graph& h) {
        const int coarsen_to = std::clamp(h.size() / 8, 40, kCoarsenTo);
        return multilevel(h, rng, max_part, coarsen_to, 1 << 30, [&](const Graph& c) {
            return initial_separator(c, rng, max_part);
        });
    };
    if (g.size() < kIntermediateAbove) { return bisect(g); }

    // Large graphs: coarsen a few levels, bisect the intermediate graph several
    // times and refine only the lightest balanced separator back to g
    const int coarsen_to = std::max(kCoarsenTo, g.size() / 30);
    return multilevel(g, rng, max_part, coarsen_to, 4, [&](const Graph& mid) {
        std::vector<signed char> best;
        std::array<int, 3>       best_state{};
        for (int run = 0; run < kIntermediateRuns; ++run) {
            std::vector<signed char> where = bisect(mid);
            std::array<int, 3>       weight{};
            for (int v = 0; v < mid.size(); ++v) { weight[where[v]] += mid.vertex_weight[v]; }
            const std::array<int, 3> state{std::max(0, std::max(weight[0], weight[1]) - max_part), weight[2], 0};
            if (best.empty() || state < best_state) {
                best       = std::move(where);
                best_state = state;
            }
        }
        return best;
    });
}

/**
 * Compute a separator through a coarse hierarchy and refine it while uncoarsening.
 *
 * The graph is coarsened by heavy-edge matching until it has at most
 * coarsen_to vertices, max_levels levels were built or matching stalls. The
 * coarse vertex weight is limited to 1.5 times the average coarsest vertex
 * weight. The supplied callback separates the coarsest graph; its partition is
 * then projected to every finer level, where a separator vertex expands to
 * both matched fine vertices, and refined by node FM moves.
 *
 * @param g Graph to separate.
 * @param rng Pseudo-random state of the matching.
 * @param max_part Maximum weight of either part.
 * @param coarsen_to Target size of the coarsest graph.
 * @param max_levels Maximum number of coarsening levels.
 * @param coarsest Separator computation on the coarsest graph.
 * @return Part label per vertex: 0 and 1 for the parts, 2 for the separator.
 */
std::vector<signed char> NestedDissection::multilevel(
    const Graph&                                                 g,
    std::uint64_t&                                               rng,
    int                                                          max_part,
    int                                                          coarsen_to,
    int                                                          max_levels,
    const std::function<std::vector<signed char>(const Graph&)>& coarsest
) const {
    // Build the coarse graph hierarchy and the fine-to-coarse vertex maps
    const int total      = std::accumulate(g.vertex_weight.begin(), g.vertex_weight.end(), 0);
    const int max_vertex = std::max(1, static_cast<int>(1.5 * total / coarsen_to));
    std::vector<Graph>            levels;
    std::vector<std::vector<int>> maps;
    levels.reserve(64);
    const Graph* current = &g;
    while (current->size() > coarsen_to && static_cast<int>(levels.size()) < max_levels) {
        std::vector<int> map;
        Graph next = coarsen(*current, map, max_vertex, rng);
        if (next.size() == 0) { break; }
        maps.push_back(std::move(map));
        levels.push_back(std::move(next));
        current = &levels.back();
    }

    // Separate the coarsest graph, then project and refine level by level
    std::vector<signed char> where = coarsest(*current);
    for (int level = static_cast<int>(levels.size()) - 1; level >= 0; --level) {
        const Graph&             fine = level == 0 ? g : levels[level - 1];
        std::vector<signed char> projected(fine.size());
        for (int v = 0; v < fine.size(); ++v) { projected[v] = where[maps[level][v]]; }
        refine(fine, projected, max_part);
        where = std::move(projected);
        levels.pop_back();
        maps.pop_back();
    }
    return where;
}

/**
 * Coarsen a graph by heavy-edge matching.
 *
 * Short contiguous vertex ranges are shuffled to retain adjacency locality.
 * Each unmatched vertex is matched with the unmatched neighbor of the heaviest
 * connecting edge, unless the merged weight would exceed max_vertex.
 * Coarse vertices are numbered in
 * increasing order of their smallest fine vertex. Parallel coarse edges are
 * merged and their weights summed; edges inside a coarse vertex vanish.
 *
 * @param g Fine graph.
 * @param map Receives the coarse vertex of every fine vertex.
 * @param max_vertex Maximum weight of a coarse vertex.
 * @param rng Pseudo-random state advanced by the visiting permutation.
 * @return Coarse graph, or an empty graph when matching cannot reduce the
 *         vertex count sufficiently. The rejected adjacency is never assembled.
 */
Graph NestedDissection::coarsen(const Graph& g, std::vector<int>& map, int max_vertex, std::uint64_t& rng) const {
    const int n = g.size();

    // Reuse scratch owned by the executing worker. No reference survives a
    // task scheduling point; coarsen itself only runs ordinary serial loops.
    int worker = 0;
#ifdef _OPENMP
    if (threads > 1) { worker = omp_get_thread_num(); }
#endif
    auto& [blocks, match, first_member, slot] = coarsen_workspace[worker];

    // Shuffle the same short vertex ranges without materializing a full visit
    // permutation. The matching order and random-generator state are unchanged.
    constexpr int visit_block = 16;
    blocks.resize(n / visit_block + (n % visit_block != 0));
    std::iota(blocks.begin(), blocks.end(), 0);
    for (int i = static_cast<int>(blocks.size()) - 1; i > 0; --i) {
        rng = mix(rng);
        std::swap(blocks[i], blocks[static_cast<int>(rng % static_cast<std::uint64_t>(i + 1))]);
    }

    // Match every vertex with its heaviest admissible unmatched neighbor
    match.assign(n, -1);
    for (int block : blocks) {
        const int first = block * visit_block;
        const int last  = first + std::min(visit_block, n - first);
        for (int v = first; v < last; ++v) {
            if (match[v] != -1) { continue; }
            int best   = v;
            int weight = 0;
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                const int u = g.adj[e];
                if (match[u] == -1 && g.edge_weight[e] > weight &&
                    g.vertex_weight[v] + g.vertex_weight[u] <= max_vertex) {
                    best   = u;
                    weight = g.edge_weight[e];
                }
            }
            match[v]    = best;
            match[best] = v;
        }
    }

    // Number coarse vertices by their smallest member
    map.assign(n, -1);
    first_member.clear();
    first_member.reserve(n);
    for (int v = 0; v < n; ++v) {
        if (map[v] != -1) { continue; }
        map[v] = map[match[v]] = static_cast<int>(first_member.size());
        first_member.push_back(v);
    }

    // Reject a stalled matching before building a graph that the hierarchy
    // would discard. Matching and RNG state are the same as for a full attempt.
    const int cn = static_cast<int>(first_member.size());
    if (cn > kCoarsenReduction * n) { return {}; }

    // Merge the adjacency of both members and accumulate parallel edge weights
    Graph coarse;
    coarse.ptr.reserve(static_cast<std::size_t>(cn) + 1);
    coarse.adj.reserve(g.adj.size());
    coarse.edge_weight.reserve(g.adj.size());
    coarse.vertex_weight.resize(cn);
    slot.assign(cn, -1);
    for (int c = 0; c < cn; ++c) {
        const int a     = first_member[c];
        const int b     = match[a];
        const int start = static_cast<int>(coarse.adj.size());
        coarse.vertex_weight[c] = g.vertex_weight[a] + (b != a ? g.vertex_weight[b] : 0);
        for (int member : {a, b}) {
            for (int e = g.ptr[member]; e < g.ptr[member + 1]; ++e) {
                const int t = map[g.adj[e]];
                if (t == c) { continue; }
                if (slot[t] >= start) {
                    coarse.edge_weight[slot[t]] += g.edge_weight[e];
                } else {
                    slot[t] = static_cast<int>(coarse.adj.size());
                    coarse.adj.push_back(t);
                    coarse.edge_weight.push_back(g.edge_weight[e]);
                }
            }
            if (b == a) { break; }
        }
        coarse.ptr.push_back(static_cast<int>(coarse.adj.size()));
    }
    return coarse;
}

/**
 * Compute the best of several grown separators of the coarsest graph.
 *
 * Every try grows part zero breadth-first from a pseudo-random seed until it
 * holds half of the total weight; the remaining vertices form part one. The
 * lighter of the two boundary layers becomes the initial vertex separator,
 * which is then improved by node FM refinement. The lightest balanced result
 * is returned.
 *
 * @param g Coarsest graph.
 * @param rng Pseudo-random state for the growing seeds.
 * @param max_part Maximum weight of either part.
 * @return Part label per vertex: 0 and 1 for the parts, 2 for the separator.
 */
std::vector<signed char> NestedDissection::initial_separator(const Graph& g, std::uint64_t& rng, int max_part) const {
    const int n     = g.size();
    const int total = std::accumulate(g.vertex_weight.begin(), g.vertex_weight.end(), 0);

    std::vector<signed char> best;
    std::array<int, 3>       best_weight{};
    std::vector<signed char> where(n);
    std::vector<char>        queued(n);
    std::vector<int>         queue;
    queue.reserve(n);
    for (int attempt = 0; attempt < kInitialTries; ++attempt) {
        // Grow part zero breadth-first until it holds half of the weight. A
        // drained queue restarts from the next unvisited vertex.
        std::fill(where.begin(), where.end(), 1);
        std::fill(queued.begin(), queued.end(), 0);
        queue.clear();
        rng = mix(rng);
        int next   = static_cast<int>(rng % static_cast<std::uint64_t>(n));
        int grown  = 0;
        int head   = 0;
        while (2 * grown < total) {
            if (head == static_cast<int>(queue.size())) {
                while (queued[next]) { next = (next + 1) % n; }
                queued[next] = 1;
                queue.push_back(next);
            }
            const int v = queue[head++];
            where[v] = 0;
            grown   += g.vertex_weight[v];
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                const int u = g.adj[e];
                if (!queued[u]) {
                    queued[u] = 1;
                    queue.push_back(u);
                }
            }
        }

        // Turn the lighter boundary layer into the initial vertex separator
        std::array<int, 2> boundary{};
        for (int v = 0; v < n; ++v) {
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                if (where[g.adj[e]] != where[v]) {
                    boundary[where[v]] += g.vertex_weight[v];
                    break;
                }
            }
        }
        const int side = boundary[0] <= boundary[1] ? 0 : 1;
        for (int v = 0; v < n; ++v) {
            if (where[v] != side) { continue; }
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                if (where[g.adj[e]] == 1 - side) {
                    where[v] = 2;
                    break;
                }
            }
        }

        // Refine the grown separator and keep the lightest balanced result
        refine(g, where, max_part);
        std::array<int, 3> weight{};
        for (int v = 0; v < n; ++v) { weight[where[v]] += g.vertex_weight[v]; }
        const int excess      = std::max(0, std::max(weight[0], weight[1]) - max_part);
        const int best_excess = std::max(0, std::max(best_weight[0], best_weight[1]) - max_part);
        if (best.empty() || excess < best_excess || (excess == best_excess && weight[2] < best_weight[2])) {
            best        = where;
            best_weight = weight;
        }
    }
    return best;
}

/**
 * Improve a vertex separator by one-sided Fiduccia-Mattheyses node moves.
 *
 * Moving separator vertex v into part `to` pulls all its neighbors from the
 * opposite part into the separator. The gain of this move is
 *
 *     gain_to(v) = w(v) - sum of w(u) over neighbors u in part 1 - to,
 *
 * i.e. the decrease of the separator weight. Every pass moves vertices into
 * one destination only, starting with the lighter part and alternating
 * between passes. Within a pass the move of largest gain is applied
 * repeatedly, including negative gains, moved vertices are locked and the
 * gains of affected separator vertices are updated incrementally. A pass ends
 * after a bounded number of moves without improvement and rolls back to its
 * best state. States are compared by balance violation, separator weight and
 * part weight difference, in this order. Refinement stops once both
 * directions failed to improve in consecutive passes.
 *
 * Only separator vertices enter the heap, and the separator after a pass is a
 * subset of the previous separator and the vertices touched by the retained
 * moves, so a pass costs time proportional to the separator neighborhood.
 *
 * @param g Graph of the partition.
 * @param where Part label per vertex, updated in place.
 * @param max_part Maximum weight of either part.
 */
void NestedDissection::refine(const Graph& g, std::vector<signed char>& where, int max_part) const {
    const int               n  = g.size();
    const std::vector<int>& vw = g.vertex_weight;

    // Accumulate the part weights and collect the initial separator
    std::array<int, 3> weight{};
    std::vector<int>   separator;
    for (int v = 0; v < n; ++v) {
        weight[where[v]] += vw[v];
        if (where[v] == 2) { separator.push_back(v); }
    }

    // State measure: balance violation, separator weight, part weight difference
    auto state = [&]() {
        return std::array<int, 3>{
            std::max(0, std::max(weight[0], weight[1]) - max_part),
            weight[2],
            std::abs(weight[0] - weight[1])
        };
    };

    // Gain of moving separator vertex v into the given part
    auto gain = [&](int v, int to) {
        int value = vw[v];
        for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
            if (where[g.adj[e]] == 1 - to) { value -= vw[g.adj[e]]; }
        }
        return value;
    };

    GainHeap heap;
    heap.resize(n);
    std::vector<int>                         locked(n, -1);
    std::vector<int>                         listed(n, -1);
    std::vector<std::pair<int, signed char>> log;
    int to   = weight[0] < weight[1] ? 1 : 0;
    int idle = 0;
    for (int pass = 0; pass < 2 * kRefinePasses && !separator.empty(); ++pass) {
        // Alternate the destination and insert the current separator vertices
        to = 1 - to;
        const int from = 1 - to;
        for (int v : separator) { heap.insert(v, gain(v, to)); }
        const std::array<int, 3> start      = state();
        std::array<int, 3>       best       = start;
        std::size_t              best_moves = 0;
        int                      stall      = 0;
        log.clear();

        while (!heap.empty()) {
            // Skip candidates that would overload the destination part
            const int v = heap.top();
            if (weight[to] + vw[v] > max_part) {
                heap.remove(v);
                continue;
            }

            // Move v into its destination and lock it for the rest of the pass
            heap.remove(v);
            locked[v] = pass;
            log.emplace_back(v, 2);
            where[v]    = static_cast<signed char>(to);
            weight[2]  -= vw[v];
            weight[to] += vw[v];

            // Pull the opposite-part neighbors into the separator. Their own
            // separator neighbors no longer pull them when moving into `to`.
            for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                const int u = g.adj[e];
                if (where[u] != from) { continue; }
                log.emplace_back(u, static_cast<signed char>(from));
                where[u]      = 2;
                weight[from] -= vw[u];
                weight[2]    += vw[u];
                for (int f = g.ptr[u]; f < g.ptr[u + 1]; ++f) {
                    const int x = g.adj[f];
                    if (where[x] == 2 && heap.contains(x)) { heap.change(x, vw[u]); }
                }
                if (locked[u] != pass) { heap.insert(u, gain(u, to)); }
            }

            // Remember the best state of the pass and stop after a long stall
            const std::array<int, 3> current = state();
            if (current < best) {
                best       = current;
                best_moves = log.size();
                stall      = 0;
            } else if (++stall > kStallMoves) {
                break;
            }
        }

        // Roll back all moves after the best state of the pass
        heap.clear();
        while (log.size() > best_moves) {
            const auto [v, previous] = log.back();
            log.pop_back();
            weight[where[v]] -= vw[v];
            weight[previous] += vw[v];
            where[v]          = previous;
        }

        // Collect the new separator from the old one and the retained moves
        std::vector<int> next;
        next.reserve(separator.size());
        auto keep = [&](int v) {
            if (where[v] == 2 && listed[v] != pass) {
                listed[v] = pass;
                next.push_back(v);
            }
        };
        for (int v : separator) { keep(v); }
        for (const auto& [v, previous] : log) { keep(v); }
        separator = std::move(next);

        // Stop after both directions failed to improve
        if (best < start) { idle = 0; } else if (++idle >= 2) { break; }
    }
}

/**
 * Build the induced subgraphs of all labels 0..count-1 in one traversal.
 *
 * Vertices with other labels, in particular separator vertices labeled -1,
 * are dropped together with all their edges. Local vertex numbers follow the
 * parent order, and edge weights are preserved.
 *
 * @param g Labeled parent graph.
 * @param ids Global identifiers of the parent vertices.
 * @param label Target subgraph per parent vertex.
 * @param count Number of target subgraphs.
 * @param parts Receives the induced subgraphs.
 * @param part_ids Receives the global identifiers of every subgraph.
 */
void NestedDissection::split(const Graph&                   g,
                             const std::vector<int>&        ids,
                             const std::vector<int>&        label,
                             int                            count,
                             std::vector<Graph>&            parts,
                             std::vector<std::vector<int>>& part_ids) const {
    // Assign local numbers and vertex data in parent order
    const int n = g.size();
    parts.assign(count, Graph{});
    part_ids.assign(count, {});
    std::vector<int> local(n, -1);
    for (int v = 0; v < n; ++v) {
        const int l = label[v];
        if (l < 0 || l >= count) { continue; }
        local[v] = static_cast<int>(part_ids[l].size());
        part_ids[l].push_back(ids[v]);
        parts[l].vertex_weight.push_back(g.vertex_weight[v]);
    }

    // Copy every edge whose endpoints share the same target subgraph
    for (int v = 0; v < n; ++v) {
        const int l = label[v];
        if (l < 0 || l >= count) { continue; }
        Graph& part = parts[l];
        for (int e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
            const int u = g.adj[e];
            if (label[u] != l) { continue; }
            part.adj.push_back(local[u]);
            part.edge_weight.push_back(g.edge_weight[e]);
        }
        part.ptr.push_back(static_cast<int>(part.adj.size()));
    }
}

/**
 * Order a small subdomain by exact minimum degree including its halo.
 *
 * The halo consists of all complete-graph neighbors outside the subdomain.
 * These vertices belong to ancestor separators and are eliminated later, so
 * they contribute to the degree of every subdomain vertex and receive fill,
 * but are never selected. The elimination graph of the subdomain is stored as
 * dense bit rows over subdomain and halo vertices.
 *
 * The weighted degree of a vertex is the scalar DOF count of its current
 * neighbors. Eliminating v connects all its remaining neighbors pairwise;
 * only the degrees of those neighbors change. Ties keep the lower local index.
 *
 * @param ids Global identifiers of the subdomain vertices.
 * @param first First output position owned by the subdomain.
 */
void NestedDissection::order_leaf(const std::vector<int>& ids, int first) {
    // Number subdomain vertices first and append their halo
    int thread = 0;
#ifdef _OPENMP
    if (threads > 1) { thread = omp_get_thread_num(); }
#endif
    std::vector<int>& local = leaf_local[thread];
    if (local.empty()) { local.assign(graph.size(), -1); }
    const int        size = static_cast<int>(ids.size());
    std::vector<int> vertices(ids);
    for (int i = 0; i < size; ++i) { local[ids[i]] = i; }
    for (int i = 0; i < size; ++i) {
        const int v = ids[i];
        for (int e = graph.ptr[v]; e < graph.ptr[v + 1]; ++e) {
            const int u = graph.adj[e];
            if (local[u] == -1) {
                local[u] = static_cast<int>(vertices.size());
                vertices.push_back(u);
            }
        }
    }
    const int total = static_cast<int>(vertices.size());
    const int words = (total + 63) / 64;

    // Build dense adjacency rows of the subdomain vertices and the live mask
    std::vector<std::uint64_t> rows(static_cast<std::size_t>(size) * words, 0);
    std::vector<std::uint64_t> alive(words, 0);
    std::vector<int>           weight(total);
    for (int i = 0; i < total; ++i) {
        alive[i / 64] |= std::uint64_t(1) << (i % 64);
        weight[i]      = graph.vertex_weight[vertices[i]];
    }
    for (int i = 0; i < size; ++i) {
        std::uint64_t* row = rows.data() + static_cast<std::size_t>(i) * words;
        const int      v   = ids[i];
        for (int e = graph.ptr[v]; e < graph.ptr[v + 1]; ++e) {
            const int u = local[graph.adj[e]];
            row[u / 64] |= std::uint64_t(1) << (u % 64);
        }
    }

    // Weighted degree over the live neighbors of one subdomain vertex
    auto degree_of = [&](int i) {
        const std::uint64_t* row   = rows.data() + static_cast<std::size_t>(i) * words;
        int                  value = 0;
        for (int w = 0; w < words; ++w) {
            for (std::uint64_t bits = row[w] & alive[w]; bits; bits &= bits - 1) {
                value += weight[w * 64 + std::countr_zero(bits)];
            }
        }
        return value;
    };
    std::vector<int> degree(size);
    for (int i = 0; i < size; ++i) { degree[i] = degree_of(i); }

    // Eliminate the vertex of minimum degree and form the clique of its neighbors
    std::vector<char>          eliminated(size, 0);
    std::vector<std::uint64_t> clique(words);
    for (int step = 0; step < size; ++step) {
        int best = -1;
        for (int i = 0; i < size; ++i) {
            if (!eliminated[i] && (best == -1 || degree[i] < degree[best])) { best = i; }
        }
        eliminated[best]          = 1;
        order[first + step]       = ids[best];
        alive[best / 64]         &= ~(std::uint64_t(1) << (best % 64));
        const std::uint64_t* row  = rows.data() + static_cast<std::size_t>(best) * words;
        for (int w = 0; w < words; ++w) { clique[w] = row[w] & alive[w]; }

        // Only subdomain neighbors are updated; halo rows are never needed
        for (int w = 0; w < words && w * 64 < size; ++w) {
            for (std::uint64_t bits = clique[w]; bits; bits &= bits - 1) {
                const int u = w * 64 + std::countr_zero(bits);
                if (u >= size) { break; }
                std::uint64_t* target = rows.data() + static_cast<std::size_t>(u) * words;
                for (int k = 0; k < words; ++k) { target[k] |= clique[k]; }
                target[u / 64] &= ~(std::uint64_t(1) << (u % 64));
                degree[u] = degree_of(u);
            }
        }
    }

    // Release the global markers for the next leaf of this thread
    for (int u : vertices) { local[u] = -1; }
}

} // namespace

/**
 * Compute a multilevel nested-dissection ordering of a symmetric pattern.
 *
 * The full scalar adjacency is formed from the lower triangle with sorted
 * rows. Unknowns u and v are indistinguishable when their closed adjacency
 * sets N(u) + {u} and N(v) + {v} are equal. Candidates are grouped by degree
 * and a sum of hashed neighbor identifiers, and every candidate pair is then
 * verified exactly, so hash collisions cannot merge unrelated unknowns.
 *
 * Consecutive unknowns whose closed neighborhoods overlap by at least
 * kSimilarity are merged as well. This captures FE nodes whose unknowns differ
 * only by individual exact zeros; any grouping still yields a valid ordering.
 *
 * Each group becomes one weighted supervariable of the compressed graph. Its
 * edges are weighted by the number of common neighbors of their endpoints, so
 * heavy-edge matching first merges strongly coupled unknowns. This matters
 * when unknowns of one node are not structurally identical, as for the
 * decoupled membrane and bending DOFs of a flat shell. Nested dissection
 * orders the compressed graph; every supervariable is then expanded into its
 * members in ascending original order, so they form a chain of fronts in the
 * elimination tree.
 *
 * @param matrix Symmetric matrix whose lower-triangular pattern is ordered.
 * @param threads Number of OpenMP threads for independent subdomains.
 * @return Permutation with order[new] = original.
 */
std::vector<int> nested_dissection(const Matrix& matrix, int threads) {
    // Validate the execution mode before building any graph
    if (threads < 1) { throw std::runtime_error("Ordering thread count must be positive"); }
#ifndef _OPENMP
    if (threads > 1) { throw std::runtime_error("Build with WITH_OPENMP=1 for parallel ordering"); }
#endif
    const int n = matrix.n;
    if (n == 0) { return {}; }

    // Build the sorted full scalar adjacency without diagonal entries. Rows of
    // earlier columns arrive first, so every adjacency row stays ascending.
    std::vector<std::size_t> ptr(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < n; ++c) {
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
            const int r = matrix.row[p];
            if (r != c) {
                ++ptr[c + 1];
                ++ptr[r + 1];
            }
        }
    }
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
    std::vector<int> adj(ptr[n]);
    {
        auto next = ptr;
        for (int c = 0; c < n; ++c) {
            for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c + 1]; ++p) {
                const int r = matrix.row[p];
                if (r != c) {
                    adj[next[c]++] = r;
                    adj[next[r]++] = c;
                }
            }
        }
    }

    // Hash every closed neighborhood and sort the unknowns into candidate runs
    std::vector<std::uint64_t> hash(n);
    for (int v = 0; v < n; ++v) {
        std::uint64_t h = mix(static_cast<std::uint64_t>(v));
        for (std::size_t p = ptr[v]; p < ptr[v + 1]; ++p) { h += mix(static_cast<std::uint64_t>(adj[p])); }
        hash[v] = h;
    }
    std::vector<int> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const std::size_t da = ptr[a + 1] - ptr[a];
        const std::size_t db = ptr[b + 1] - ptr[b];
        if (da != db) { return da < db; }
        if (hash[a] != hash[b]) { return hash[a] < hash[b]; }
        return a < b;
    });

    // Verify closed-neighborhood equality inside every run. The first
    // unassigned member of a run is the smallest unknown of its group.
    std::vector<int> representative(n, -1);
    std::vector<int> marker(n, -1);
    for (int begin = 0; begin < n;) {
        int end = begin + 1;
        while (end < n && hash[candidates[end]] == hash[candidates[begin]] &&
               ptr[candidates[end] + 1] - ptr[candidates[end]] == ptr[candidates[begin] + 1] - ptr[candidates[begin]]) {
            ++end;
        }
        for (int i = begin; i < end; ++i) {
            const int v = candidates[i];
            if (representative[v] != -1) { continue; }
            representative[v] = v;
            if (end - begin == 1) { continue; }
            marker[v] = v;
            for (std::size_t p = ptr[v]; p < ptr[v + 1]; ++p) { marker[adj[p]] = v; }
            for (int j = i + 1; j < end; ++j) {
                const int w = candidates[j];
                if (representative[w] != -1 || marker[w] != v) { continue; }
                bool equal = true;
                for (std::size_t p = ptr[w]; p < ptr[w + 1] && equal; ++p) { equal = marker[adj[p]] == v; }
                if (equal) { representative[w] = v; }
            }
        }
        begin = end;
    }

    // Merge consecutive unknowns whose closed neighborhoods are nearly equal.
    // Exact numerical zeros removed at import make the patterns of one FE
    // node differ slightly; adjacent mesh nodes overlap far less, so a high
    // similarity threshold separates both cases. Each candidate is compared
    // with the smallest member of the current group to prevent drift along
    // a chain of unknowns.
    auto find = [&](int v) {
        while (representative[v] != v) {
            representative[v] = representative[representative[v]];
            v                 = representative[v];
        }
        return v;
    };
    for (int b = 1; b < n; ++b) {
        const int a = find(b - 1);
        const int c = find(b);
        if (a == c) { continue; }
        const auto a_begin = adj.begin() + ptr[a];
        const auto a_end   = adj.begin() + ptr[a + 1];
        if (!std::binary_search(a_begin, a_end, b)) { continue; }
        std::size_t common = 0;
        for (auto p = a_begin, q = adj.begin() + ptr[b], q_end = adj.begin() + ptr[b + 1]; p != a_end && q != q_end;) {
            if (*p < *q) { ++p; } else if (*q < *p) { ++q; } else { ++common; ++p; ++q; }
        }
        const std::size_t shared = common + 2;
        const std::size_t joint  = (ptr[a + 1] - ptr[a] + 1) + (ptr[b + 1] - ptr[b] + 1) - shared;
        if (static_cast<double>(shared) >= kSimilarity * static_cast<double>(joint)) {
            representative[std::max(a, c)] = std::min(a, c);
        }
    }

    // Number supervariables by their smallest member and list members ascending
    std::vector<int> super(n);
    std::vector<int> member_ptr{0};
    for (int v = 0; v < n; ++v) {
        const int r = find(v);
        if (r == v) {
            super[v] = static_cast<int>(member_ptr.size()) - 1;
            member_ptr.push_back(0);
        } else {
            super[v] = super[r];
        }
        ++member_ptr[super[v] + 1];
    }
    const int compressed = static_cast<int>(member_ptr.size()) - 1;
    std::partial_sum(member_ptr.begin(), member_ptr.end(), member_ptr.begin());
    std::vector<int> members(n);
    {
        auto next = member_ptr;
        for (int v = 0; v < n; ++v) { members[next[super[v]]++] = v; }
    }

    // Build the compressed graph from the union of the member adjacencies.
    // Nearly equal members contribute only a few additional edges.
    Graph graph;
    graph.ptr.reserve(static_cast<std::size_t>(compressed) + 1);
    graph.vertex_weight.resize(compressed);
    std::fill(marker.begin(), marker.begin() + compressed, -1);
    for (int s = 0; s < compressed; ++s) {
        graph.vertex_weight[s] = member_ptr[s + 1] - member_ptr[s];
        for (int m = member_ptr[s]; m < member_ptr[s + 1]; ++m) {
            const int v = members[m];
            for (std::size_t p = ptr[v]; p < ptr[v + 1]; ++p) {
                const int t = super[adj[p]];
                if (t != s && marker[t] != s) {
                    marker[t] = s;
                    graph.adj.push_back(t);
                }
            }
        }
        graph.ptr.push_back(static_cast<int>(graph.adj.size()));
    }
    graph.edge_weight.assign(graph.adj.size(), 1);

    // Weight every edge by the number of common neighbors of its endpoints.
    // Unknowns of one mesh node share almost all neighbors, so heavy-edge
    // matching merges them first and the coarse graphs keep the mesh geometry.
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        std::vector<int> mark(compressed, -1);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
        for (int v = 0; v < compressed; ++v) {
            for (int e = graph.ptr[v]; e < graph.ptr[v + 1]; ++e) { mark[graph.adj[e]] = v; }
            for (int e = graph.ptr[v]; e < graph.ptr[v + 1]; ++e) {
                const int u      = graph.adj[e];
                int       common = 0;
                for (int f = graph.ptr[u]; f < graph.ptr[u + 1]; ++f) { common += mark[graph.adj[f]] == v; }
                graph.edge_weight[e] = 1 + common;
            }
        }
    }
    adj  = {};
    hash = {};

    // Dissect the compressed graph, concurrently for large independent parts
    NestedDissection nd{
        graph,
        std::vector<int>(compressed, -1),
        std::vector<std::vector<int>>(threads),
        std::vector<std::array<std::vector<int>, 4>>(threads),
        threads
    };
    std::vector<int> ids(compressed);
    std::iota(ids.begin(), ids.end(), 0);
#ifdef _OPENMP
    if (threads > 1) {
#pragma omp parallel num_threads(threads)
#pragma omp single
        nd.dissect(graph, ids, 0);
    } else {
        nd.dissect(graph, ids, 0);
    }
#else
    nd.dissect(graph, ids, 0);
#endif

    // Expand every supervariable into its consecutive scalar members
    std::vector<int> order;
    order.reserve(n);
    for (int s : nd.order) {
        order.insert(order.end(), members.begin() + member_ptr[s], members.begin() + member_ptr[s + 1]);
    }
    return order;
}

} // namespace mf
