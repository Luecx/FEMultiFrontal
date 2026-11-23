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
