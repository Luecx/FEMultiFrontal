# Experimental signed LDLT batch panels

This experiment adds a **strictly opt-in** multi-pivot front-local fast path for
symmetric pivoted LDLT. The pre-existing 1x1/2x2 threshold-pivoted path,
inertia/rank estimation and packed Schur update remain the fallback. It is
**not** an SPD/Cholesky conversion and it never regularizes a pivot.

## Algorithm

At an empty numerical panel, gather the first `k` fully summed eligible
columns of the assembled frontal matrix into one row-major `m x k`
workspace (`m` current active variables, `k<=min(eligible,block_size-1,width)`).
The workspace contains the triangular candidate block and all couplings
to the active update region.

Try the candidate prefix in its original active order. Before accepting each
1x1 signed diagonal `d`, the workspace has already received all preceding
signed rank-one corrections. Require finite entries, `abs(d)>rank_tolerance`
and the **same full-current-active-column threshold**
`abs(d)>=pivot_tolerance*max(abs(C[j:m,j]))` as the existing solver.
Negative diagonals are permitted. On success, append the signed D entry,
normalized multipliers, and one left/right Schur column to the existing
factor/panel storage. Update only the remaining narrow workspace columns.
An unstable, zero or near-zero candidate immediately ends the batch.

Remove only the accepted pivot prefix, update cached diagonals in the
original pivot order, and resume ordinary robust 1x1/2x2 pivot selection.
A batch never crosses a numerical panel boundary and never delays or removes
an unaccepted pivot. The physical packed front is updated only by the
existing final rank-k Schur kernel.

This initial prototype prioritizes correctness and separable profiling over
peak speed. In particular, narrow workspace gather and sequential rank-one
updates may cost more than the cache they replace.

## Controls

- `--ldlt-signed-batch-panels`: enable experiment; **off by default**.
- `--ldlt-signed-batch-width N`: width of the trial panel (4..256; default 16).
- `signed_batch_attempts` / `signed_batch_pivots`: diagnostics in the
  standard solver summary.

The experiment applies only to blocked symmetric LDLT with fast pivoting
enabled and `block_size >= 5`. Scalar LDLT and LU are unchanged.

## Warp9 paired benchmark

```bash
git fetch origin
git switch perf/ldlt-signed-batch-panels && git pull --ff-only
make -B -j8 WITH_OPENMP=0 all check \
  CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/haken.mtx.zst --binary build/native/mfsolve \
  --threads 1 --ordering nd --cpu 90 --rounds 8 --warmups 2 \
  --signed-batch-ablation --signed-batch-width 16

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/connector_2_quad.mtx.zst --binary build/native/mfsolve \
  --threads 1 --ordering nd --cpu 90 --rounds 8 --warmups 2 \
  --signed-batch-ablation --signed-batch-width 16
```

The paired harness enables both previously merged cache optimizations in
**both** variants and toggles only the signed batch experiment. It checks
equal numerical rank and inertia as well as independent original-system
residuals. Try `--signed-batch-width 8`, `16`, `24` and `30` as separate
controlled A/B experiments, not by changing supernode width simultaneously.

**Do not merge** before numerical CI is green, both matrices are benchmarked,
and a sustained end-to-end speedup is demonstrated. A larger fast-pivot count
alone does not establish an optimization.
