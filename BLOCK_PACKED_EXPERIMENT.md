# Experimental block-packed triangle backend

This branch introduces an **opt-in** 48x48 block-packed lower-triangle layout for temporary SPD multifrontal contributions. The production `src/factor.cpp` remains unchanged; `src/factor_bpt.cpp` is selected only by `BPT=1`.

## Build and compare

```bash
make -j8 WITH_OPENMP=1 check
make -j8 WITH_OPENMP=1 BPT=1 check

make -j8 WITH_OPENMP=1 all
make -j8 WITH_OPENMP=1 BPT=1 all
./build/native-omp/mfsolve matrix.mtx --repeat 5 --threads 1
./build/native-omp-bpt/mfsolve matrix.mtx --repeat 5 --threads 1
```

The two configurations use separate object directories. Run both on the same matrix, with identical settings and CPU affinity. Compare numerical factorization and total time separately.

## Scope

- Contiguous, row-major dense lower-block tiles; full square diagonal tiles.
- Same scalar CSC factors and solve implementation as the baseline.
- Origin-offset inheritance, child extend-add, panel assembly and in-place Schur updates adapted to the new storage.
- `tests/bpt.cpp` compares BPT kernels with a scalar packed-triangle reference for several sizes, offsets and gaps.

## Limitations

**Experimental; not yet full-suite validated or benchmarked in this environment.** Isolated BPT kernel tests were performed in the prototype before publishing this branch. A complete repository checkout was unavailable in the execution environment.

The `at(i,j)` scalar addressing in extend-add and panel assembly is likely slower on small fronts, and the block-packed Schur kernel is not yet a tuned GEMM implementation. Do not assume BPT is faster until production FEM benchmarks confirm it.
