# Fine-grained multifrontal LDLT profiling

Set `FEMF_LDLT_PROFILE=1` to accumulate measurements for each front.
The feature is **opt-in** and does not change the numerical operations,
pivot policy, front structure, solver output or factorization format.

```bash
git fetch origin
git switch perf/ldlt-detailed-profile
make -j8 WITH_OPENMP=0 all check \
    CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

zstd -dc benchmark/indefinite/haken.mtx.zst > /tmp/haken-profile.mtx

FEMF_LDLT_PROFILE=1 taskset -c 90 \
  ./build/native/mfsolve /tmp/haken-profile.mtx \
  --matrix-type symmetric --ordering nd --threads 1 \
  --ldlt-block-size 32 --repeat 3 \
  2>&1 | tee /tmp/ldlt-profile-detailed.txt
```

All original `FEMF_LDLT_PROFILE` keys remain. New fields:

| Key | What is counted |
|---|---|
| `front_setup_s` | Initialization of active indices, L/R panel memory and pivot lookahead cache |
| `cache_refresh_s` | Refreshing fully summed candidate rows before the search |
| `cache_apply_s` | Updating cached diagonals and rows for newly accepted pivots |
| `panel_build_s` | Creating scalar/2×2 pivot coefficients, L/R panel columns and recording nonzero multipliers |
| `pivot_bookkeeping_s` | Removing accepted active indices, invalidating cache slots and finalizing pivot records |
| `contribution_s` | Building/inheriting a parent's packed contribution, or checking the unresolved root block |
| `unattributed_front_s` | `total_front_s` minus **all disjoint front phase bins** (unprofiled traversal, flush cleanup, residual time and instrumentation overhead) |
| `factor_pack_s` | Final concatenation of each front's pivot/multiplier records into the global factor **after the front traversal** |

Existing fields `assembly_s`, `pivot_search_s`, `schur_s`, and `total_front_s`
retain their original meanings. `factor_pack_s` is **outside**
`total_front_s`; do not sum it into `unattributed_front_s`.
A single thread is recommended for meaningful elapsed-time decomposition.
On an OpenMP run, front phases are accumulated over concurrently executed
fronts and are **not** directly comparable with factorization wall time.

The primary goal is to identify the next algorithmic target after the
packed-indexed Schur optimization, not to assume that all remaining time
belongs to pivot search. For `haken`, the previous profiling showed roughly
4.76 s in Schur, 2.24 s in Assembly, 0.30 s in pivot search and
3.87 s in otherwise unattributed front operations.

## Assembly and contribution subphases (added after first profiling results)

To distinguish the expensive parts of the two memory-heavy bins, the detailed
profile now also reports:

| Field | Meaning |
|---|---|
| `front_prepare_s` | Front-row union, allocation, position map and ownership reuse |
| `original_assemble_s` | Inserting scaled original matrix entries into the front |
| `extend_add_s` | Scattering child contributions into the current front |
| `contribution_copy_s` | Allocating and compacting the outgoing packed triangular contribution |
| `contribution_move_s` | Transferring reusable contiguous contribution storage without copying |
| `contribution_copy_entries` | Number of `double` entries materialized into compacted contribution buffers |
| `contribution_moved_fronts` | Number of outgoing fronts whose backing storage was inherited without copying |

These are **subphases already contained in** `assembly_s` and `contribution_s`.
Do not add them to the earlier top-level categories or to
`unattributed_front_s`. `contribution_copy_entries * 8` estimates payload
bytes written but **not** total traffic, since packed source reads, allocations
and memory writes also contribute. For a representative single-thread run,
compare the contribution copy count with the measured copy time to decide
whether to optimize physical triangular compaction or front reuse.

## Front-row union A/B experiment

Profiling on `haken` showed ~1.60 s in `front_prepare_s`, but this combines
row-pattern construction and **mandatory** dense packed-front zero-initialization.
The fast front-preparation variant eliminates repeated appends and a global
sort/unique by incrementally merging sorted symbolic and contribution row lists:

- Start with the bottom symbolic column once (previous code appended it twice).
- Skip nested member or child lists already contained in the current union.
- Merge only genuinely additional sorted lists with `std::set_union`.
- Continue to zero-fill the entire packed triangular matrix so all original
  assembly and Schur operations have the same initialized input.
- Retain the exact old concatenation/sort/unique behind
  `--ldlt-no-fast-front-prepare`, which can be evaluated with
  `benchmarks/compare_ldlt_fastpath.py --front-prepare-ablation`.

Additional **subphases** within `front_prepare_s`:

| Field | Meaning |
|---|---|
| `front_row_union_s` | Front row union construction or inherited-front move |
| `front_zero_s` | New packed front allocation and zero-initialization |
| `front_position_map_s` | Construction of row-to-local-index mapping |
| `front_contained_sources` | Number of sorted source lists skipped because all entries already exist |
| `front_merged_sources` | Number of source lists that required a union |

The nested subphases must **not** be added again to the top-level phases. A
remaining difference between `front_prepare_s` and these subphases includes
front metadata and timer overhead. If `front_zero_s` dominates, the row-union
change may deliver limited speedup; do not optimize zero initialization without
maintaining the same all-zero numerical semantics.

## Demand-driven pivot lookahead cache (A/B experiment)

The original `refresh_candidates` eagerly materialized **all eight**
eligible candidate rows and then maintained each cached row incrementally
after every accepted pivot. On the `haken` profile, these operations took
roughly 0.9 seconds of cache refresh and 0.5 seconds of cache application.

The new `lazy_pivot_cache` policy (default on, blocked LDLT only):
- Start each pivot search with only the first eligible candidate materialized.
- Keep any other already-cached row among the first eight candidates.
  **Do not** discard a delayed candidate merely because it was not tested on
  the immediately preceding iteration.
- If candidate 0 fails the threshold criterion, construct candidate 1 on
  demand, then candidate 2 only if 1 also fails, and so on.
- Preserve the old full lookahead and cached diagonal update equations,
  `raw_current` implementation, fallback exhaustive threshold search,
  2x2 pivot selection, the Schur update and all pivot-acceptance thresholds.

For comparison, use
`--ldlt-no-lazy-pivot-cache` with `mfsolve`, or the
`--lazy-cache-ablation` mode of `compare_ldlt_fastpath.py`.
The latter uses the same binary and alternating A/B rounds. The entire old
eight-row policy is available; disabling the full pivot cache is a separate
`--ldlt-no-pivot-cache` ablation.

`cache_prefetched_rows` and `cache_demand_rows` count newly computed
candidate rows. Their sum is the number of constructed cache rows. With
profiling enabled, on-demand row construction is charged to
`cache_refresh_s` **and explicitly removed from** `pivot_search_s`:
the phase timers remain nonoverlapping.

Expected advantage: substantial reduction in cached row initialization and
incremental rank-one cache updates when most pivots accept their first
candidate. The number of 2x2 pivots, numerical rank, inertia and packed
multiplier/pivot data must remain unchanged. The A/B performance comparison
on `haken` and a smaller structurally different matrix is required before
merge; clean CI by itself does not establish speedup.
