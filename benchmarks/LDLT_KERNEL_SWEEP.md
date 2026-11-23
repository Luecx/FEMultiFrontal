# LDLT Schur microkernel sweep

This is a **standalone numerical microbenchmark** for the real packed-front
operation used by FEMultiFrontal's indefinite solver:

```text
C(active,active) -= L(active,:) * (L*D)(active,:)^T
```

Unlike Cholesky, `D` may contain **negative 1x1 pivots and indefinite 2x2
blocks**. The generated right operand uses exactly these signed blocks, rather
than pretending the update is `LL^T`.

The fixture retains the physical packed lower-triangular layout, inherited
`origin`, and the `active` index holes created by delayed pivots. Every
single candidate must match an independent scalar reference at **every
allocated packed entry**, including untouched padding and inactive entries.

## Build and verify

On Warp9, use the same compiler/flags as the production `native` solver:

```bash
make kernel-sweep WITH_OPENMP=0 \
    CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

./build/native/ldlt-kernel-sweep --verify-only \
    --sizes 9,33,65,129,256 --ranks 1,2,8,32 --holes 0,12 --origins 0,19
```

## Performance sweep

```bash
taskset -c 90 ./build/native/ldlt-kernel-sweep \
    --sizes 256,768,1536 --ranks 16,32 \
    --holes 0,12 --origins 19 \
    --modes hot,cold --rounds 5 --target-ms 40 --cold-mib 128 \
    > /tmp/ldlt-kernel-sweep.csv
```

The CSV records **26 variants** with median milliseconds per update and
speedup against the independent scalar reference. Progress and correctness
messages go to stderr. The default cold pool is deliberately much larger than
a typical single-chiplet cache, but the appropriate size depends on the exact
processor. No `std::memcpy` of the output front is performed inside the timed
region; input packing done *by the kernel* is included.

### Variants

- Scalar reference and scalar column unroll factors 2, 4, 8, 16.
- **Exact production `rank_k_ldlt_tiled` kernel**, linked from the solver source.
- Direct AVX2/FMA output tiles: 4×8, 6×8, 8×8, 4×16, 6×16, 8×16,
  12×8, 4×32.
- Rank-major **B-packed** tiles: 4×8, 6×8, 6×16.
- Rank-major **A+B-packed** tiles: 4×8, 6×8, 6×16.
- **Index-hole-aware A+B-packed** tiles, without a whole-front copy:
  AVX2 gather/scalar load into register C tiles, FMA on packed operands,
  individual scatter back to active packed-front entries. Compare both
  load mechanisms at 4×8, 6×8 and 4×16.

The last group adapts the LLT `rank_k_subtract_packed6` **factor-panel
structure** to *separate signed LDLT operands*, without copying the entire
front into a transient dense matrix. The direct variants read the original
pivot-major operands. Kernels requiring contiguous columns safely fall back
to scalar updates for irregular `active` blocks.

### Interpreting results

- **Hot** repeatedly updates the same packed contribution, showing the
  processor's compute/L1/L2/LLC behavior once warmed.
- **Cold** rotates through independent packed contributions **and L/R factor
  panels**. The buffer pool attempts to exceed the effective LLC and better
  exposes memory bandwidth. Factor panels stay reused *within* each front.
- Both modes keep a *single thread*. Comparing 20 variants inside the same
  binary avoids compilation and ordering differences.
- The test excludes assembly-tree traversal, pivot selection, factor-row
  updates, Extend-Add and solution, so **the fastest kernel in isolation is
  not necessarily the fastest multifrontal solver**.
- Especially inspect `holes=12`: numerical pivot delays interrupt SIMD
  contiguity. Also repeat with the actual front-size/panel-rank distribution
  from `FEMF_LDLT_PROFILE=1` rather than tuning only the largest front.
- A winner must be confirmed in an opt-in production dispatch with A/B
  against `main` on `haken`, `connector_2_quad` and other indefinite
  matrices; do not merge solely on synthetic gigaflops.
