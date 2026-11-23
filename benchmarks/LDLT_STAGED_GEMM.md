# Experimental dense-staged GEMM for indefinite LDLT Schur updates

The production indefinite multifrontal LDLT Schur kernel updates the **row-packed
lower triangular** frontal buffer in place using a 4x8 AVX2 microtile. It
already packs its L/R operands. Replacing the entire frontal storage with a
persistent 32x32 tile-major layout was substantially slower in Warp9 testing
(PR #35, approximately 12.19 s versus 8.59 s on haken).

This separate experiment keeps **every frontal, pivot, cache, contribution,
extend-add and ownership transfer** unchanged. Only sufficiently large
Schur panel flushes opt into an alternative BLAS-3-style output staging path.

## Algorithm

For `active.size() >= 192` and accumulated pivot rank `>=16`:
1. Pack `A(i,p)=left(p,active[i])` and
   `B(p,j)=right(p,active[j])` once per panel flush.
   The original signed/indefinite 1x1/2x2 D coefficients are already
   included in the panel's right operand; do not assume SPD.
2. For each 64x64 lower-triangular output tile, gather the corresponding
   entries from the **unchanged** physically row-packed front into dense
   row-major scratch C. Holes from delayed pivots are allowed.
3. Compute `C <- C - A*B` with `cblas_dgemm` when explicitly compiled with
   `WITH_CBLAS=1`, or a portable internal blocked AVX2 4x8 GEMM kernel
   when built in standard standalone mode.
4. Check finiteness of meaningful lower-triangle results and scatter
   those entries to the original packed frontal storage. Upper-diagonal
   scratch entries are never copied into the front.

Small/low-rank updates keep the original fast in-place kernel; no extra
allocations are required in the default production path.

**Production default unchanged:** `--ldlt-gemm-schur` is opt-in and
`PivotFactor::staged_gemm_schur` defaults to `false`.
Supernode/panel defaults stay 64/32.

## Standard standalone build, correctness and paired Warp9 benchmark

```bash
git fetch origin
git switch perf/ldlt-staged-gemm-schur
git pull --ff-only
make -B -j8 WITH_OPENMP=0 all check \
  CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/haken.mtx.zst --binary build/native/mfsolve \
  --threads 1 --ordering nd --rounds 8 --warmups 2 --gemm-ablation

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/connector_2_quad.mtx.zst --binary build/native/mfsolve \
  --threads 1 --ordering nd --rounds 8 --warmups 2 --gemm-ablation
```

The comparison uses **one binary and identical settings**; the only
difference is `--ldlt-gemm-schur`. The benchmark independently checks
the original-matrix residual and matches rank, nullity and inertia.
`factor_s` excludes input parsing, permutation and symbolic analysis.

## Optional external optimized BLAS-3 (OpenBLAS or Intel MKL)

The standalone build has **no BLAS dependency**. For an optional true
vendor GEMM path, link against a CBLAS provider:

```bash
# OpenBLAS example when headers/library are on compiler search paths
make -B -j8 WITH_CBLAS=1 WITH_OPENMP=0 all check \
  BLAS_LIBS="-lopenblas" \
  CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/haken.mtx.zst \
  --binary build/native-cblas/mfsolve \
  --threads 1 --ordering nd --rounds 8 --warmups 2 --gemm-ablation

# Alternatively, with an MKL environment configured, supply your own
# headers and runtime link flags; no assumption about MKLROOT is made.
# BLAS_CPPFLAGS="-I<path-to-MKL-headers>" BLAS_LIBS="-L... -lmkl_rt ..."
```

The Makefile isolates CBLAS object files in `build/native-cblas`, to
avoid mixing compiled variants. Ensure `MKL_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` for single-core benchmarks. Using OpenMP and
internally multithreaded BLAS simultaneously may oversubscribe cores.

To profile which part of GEMM staging dominates:

```bash
zstd -dc benchmark/indefinite/haken.mtx.zst >/tmp/haken-gemm.mtx
FEMF_LDLT_PROFILE=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 90 ./build/native/mfsolve /tmp/haken-gemm.mtx \
  --matrix-type symmetric --ordering nd --threads 1 \
  --ldlt-gemm-schur --repeat 2
```

New `FEMF_LDLT_PROFILE` fields are subphases of `schur_s`:
`staged_gemm_pack_s`, `staged_gemm_load_s`,
`staged_gemm_compute_s`, `staged_gemm_store_s` and
`staged_gemm_tiles`. Do not sum them again into factorization totals.
They let us distinguish inefficient arithmetic from C gather/scatter
and repeated panel packing.

**Merge policy:** CI correctness alone is insufficient. Only merge
after haken's paired Warp9 factorization time improves materially
without an unacceptable connector regression. If copy traffic wins
over arithmetic locality, close this experiment and retain the current
in-place packed kernel.
