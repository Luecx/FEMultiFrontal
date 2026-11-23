# LDLT: isolate supernode amalgamation from Schur-panel rank

The existing `--ldlt-block-size N` controls **two independent operations**:
(1) how many consecutive elimination-tree owners may be amalgamated into
a supernode (formerly `while(count < factor.block_size)`),
and (2) the maximum numerical pivot-panel rank before a Schur flush.

Consequently, the previous 16/32/64/96/128 block-size sweep did **not**
isolate the performance of bigger Schur panels. It also changed the
supernode grouping and therefore `delayed_candidates` and front traffic.

The solver has independent settings: `--ldlt-block-size N` controls the
numerical panel rank and `--ldlt-supernode-size N` controls the maximum
supernode amalgamation width. The current tuned defaults are **32 and 64**,
respectively, for both `mfsolve` and the C++ `PivotFactor` API.

The explicit `--ldlt-supernode-size 0` (or `PivotFactor::supernode_size=0`)
restores the original coupled policy, where the supernode limit follows
`block_size`. The scalar LDLT path is unaffected.

## Warp9: 2×2 factorial A/B (fixed compiler, matrix, CPU)

```bash
git fetch origin && git switch main && git pull --ff-only
make -B -j8 WITH_OPENMP=0 all check \
  CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"
taskset -c 90 python3 benchmarks/sweep_ldlt_panel_supernode.py \
  benchmark/indefinite/haken.mtx.zst \
  --binary build/native/mfsolve \
  --pairs 32:32,32:64,64:32,64:64 --rounds 4 --warmups 1 --cpu 90
```

Each `P:S` pair sets `--ldlt-block-size P` and
`--ldlt-supernode-size S`. All runs use the same indexed 4×8 SIMD
kernel, numerical pivoting and threads. Rotating per-round run order
reduces but cannot eliminate load/thermal drift. Results print per-run
`factor_s`, median and median absolute deviation. The benchmark
independently verifies the original-system residual and requires equal
rank/nullity/inertia for all runs.

Interpret factorial contrasts:
- **32:32 vs 64:32** isolates larger numerical panels while holding
  supernode grouping fixed at 32.
- **32:32 vs 32:64** isolates larger amalgamated supernodes while
  keeping the numerical panel rank 32.
- **64:64** tests the coupled setting measured in the original sweep.

This is a diagnostic PR, **not yet a speed optimization**.
Keep `main` default unchanged until actual Warp9 results identify
a worthwhile independent parameter setting.

## Tuned default on Warp9 (2026-11-23)

The independent 32:64 setting consistently outperformed the old 32:32
coupled default on the 106,601-DOF haken matrix across two interleaved
factorial sweeps and also helped connector_2_quad. In the eight-round
sweep the medians were 9.9674 s (32:32) versus 8.9005 s (32:64) on
haken, and 0.18761 s versus 0.18464 s on connector_2_quad. All settings
preserved numerical rank, inertia and small original-system residuals.

**Default selection:** panel=32, supernode=64. Use
`--ldlt-supernode-size 0` to reproduce legacy coupled-width behavior;
with panel=32, this means 32:32. This is an empirically tuned default,
not a universal optimum for all matrix structures and platforms.
