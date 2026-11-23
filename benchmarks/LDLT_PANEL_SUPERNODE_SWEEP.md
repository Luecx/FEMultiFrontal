# LDLT: isolate supernode amalgamation from Schur-panel rank

The existing `--ldlt-block-size N` controls **two independent operations**:
(1) how many consecutive elimination-tree owners may be amalgamated into
a supernode (formerly `while(count < factor.block_size)`),
and (2) the maximum numerical pivot-panel rank before a Schur flush.

Consequently, the previous 16/32/64/96/128 block-size sweep did **not**
isolate the performance of bigger Schur panels. It also changed the
supernode grouping and therefore `delayed_candidates` and front traffic.

This branch adds an optional `--ldlt-supernode-size N` (1..256).
If omitted, supernode size follows `--ldlt-block-size`, retaining the
legacy behavior and bit-identical factors. The C++ API exposes
`PivotFactor::supernode_size`, default 0 (=follow `block_size`).
The scalar path is unchanged.

## Warp9: 2×2 factorial A/B (fixed compiler, matrix, CPU)

```bash
git fetch origin && git switch perf/ldlt-decouple-panel-supernode && git pull --ff-only
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
