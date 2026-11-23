# oneMKL PARDISO reference for symmetric-indefinite FEMF matrices

This is a **separate optional reference executable**, not part of FEMF's
required build or numerical path. It benchmarks Intel oneMKL PARDISO's
symmetric-indefinite `mtype = -2` solver (which may perturb tiny pivots),
on the same original MatrixMarket operator as FEMF.

## Build and run on Warp9

Use the already installed MKL. The script finds `mkl.h` and
`libmkl_rt.so` under `MKLROOT`, the active Conda environment or the
standard oneAPI installation path. It only compiles the reference program.

```bash
bash benchmarks/build_mkl_pardiso.sh

# MatrixMarket files must be decompressed; oneMKL benchmark does not parse zstd.
zstd -dc benchmark/indefinite/haken.mtx.zst >/tmp/haken-profile.mtx

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE taskset -c 90 \
  bash benchmarks/build_mkl_pardiso.sh /tmp/haken-profile.mtx \
    --threads 1 --warmups 2 --repeat 6
```

For `connector_2_quad` replace the input with the decompressed connector matrix.

## Interpretation and limitations

- `load_convert_s`: reading input and preparing PARDISO's **upper** CSR
  (the source lower CSC has exactly this structure by symmetry), **not**
  part of numerical factorization.
- `analysis_s`: PARDISO's own ordering and symbolic analysis (`phase=11`),
  performed once.
- `factor_s`: numerical factorization only (`phase=22`), repeated on
  the **same analysis**; compare to FEMF's `trial factor_s`.
- `solve_s`: substitution **including MKL's iterative refinement**
  (`phase=33`); compare separately.
- `relative_residual`: independent original-system A*x-b norm;
  `residual_ok=0` invalidates any accuracy-matched speed comparison.
- `perturbed_pivots` (`iparm[13]`): the number of tiny pivots PARDISO
  perturbed in order to complete factorization. FEMF instead has an
  explicit rank-aware policy and may report a nonzero nullity.
- `inertia_positive`/`inertia_negative`: PARDISO's inertia of its
  regularized numerical factorization is **not guaranteed to match**
  FEMF's rank-aware inertia for singular matrices.

The analysis is PARDISO's **own** fill-reducing ordering, not FEMF's
nested dissection. The numerical phase comparison answers which complete
solver implementation is faster *after its own symbolic analysis*, but does
not isolate dense-kernel speed with the same sparse factor pattern.

PARDISO default symmetric-indefinite pivoting and perturbation are
retained (except explicit CSR base-0 indexing, user-selected threads and
optional perturb exponent). `--matching` enables scaling + weighted
matching as a separate configuration; if you use it, report it explicitly.

The generated RHS is the same deterministic manufactured RHS as FEMF's
pivot driver, so it is consistent even when the matrix is rank deficient.
The solver result is checked against the original matrix, outside the
timed factor and solve intervals.

Reference: Intel oneMKL PARDISO C documentation
https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso.html
