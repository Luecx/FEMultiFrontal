# Schur update benchmarks

The reference revision is `50f1d8802ee8fce4cbcc22f914074fa21685653e`.
The measured results and tuning recommendation are in [warp9_results.md](warp9_results.md).
The default test is a clamped, unit-spaced C3D8 steel beam: 576 x 23 x 23
elements, 997,056 total displacement DOFs and 995,328 free DOFs.
Use FEMaster's reduced `*_A.mtx` matrix, since the unconstrained `*_K.mtx`
contains rigid-body modes. No diagonal shift or synthetic stiffness is added.

Generate the deck, then run FEMaster in the deck's directory:

```sh
python benchmarks/generate_cantilever.py dev/cantilever_1m.inp
cd dev
/path/to/FEMaster cantilever_1m.inp --ncpus 4
cd ..
```

The export basename is uppercase because FEMaster normalizes keyword values.
Matrix generation, assembly, text export and import are outside numeric timings.

Build the solver and the batch-width experiment:

```sh
make -j4 WITH_OPENMP=1 CXX=g++ CPPFLAGS=-Isrc \
    CXXFLAGS='-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp' \
    LDFLAGS=-fopenmp all check
g++ -O3 -march=native -DNDEBUG -std=c++20 -fopenmp -Isrc \
    benchmarks/block_sweep.cpp \
    $(find build/native-omp -maxdepth 1 -name '*.o' ! -name main.o) \
    -o dev/block_sweep
```

Build the optional MKL reference using the same matrix importer:

```sh
g++ -O3 -march=native -DNDEBUG -std=c++20 -Isrc \
    -I"$MKLROOT/include" benchmarks/mkl_reference.cpp \
    build/native-omp/matrix.o build/native-omp/memory.o \
    -L"$MKLROOT/lib" -Wl,-rpath,"$MKLROOT/lib" \
    -lmkl_rt -lpthread -lm -ldl -o dev/mkl_reference
```

Run these sequentially, with the same thread count, to avoid CPU contention:

```sh
dev/block_sweep dev/CANTILEVER_1M_A.mtx 1 4 profile
dev/mkl_reference dev/CANTILEVER_1M_A.mtx 1 4
dev/block_sweep dev/CANTILEVER_1M_A.mtx 4 4
dev/mkl_reference dev/CANTILEVER_1M_A.mtx 4 4
```

`block_sweep` compares the unchanged solver's default width 32 against widths
64, 128 and 256. It imports, orders and analyzes once, reuses factor capacity,
and rotates the order of widths between trials. Discard trial 1 for each width
and report the median of the remaining three trials. The optional `profile`
argument adds serial phase timers, including `schur_s`; parallel runs measure
wall time without overlapping phase timers.

MKL uses Pardiso's SPD mode, its own fill-reducing ordering, phase 11 for
analysis, phase 22 for each fresh numeric factorization and phase 33 for each
solve. Discard its first numeric trial too. Factorization and solve times are
reported separately. Include the symbolic preparation time and factor fill
when comparing complete one-shot solves, since the two solvers use different
orderings.

Both drivers manufacture the same nonconstant solution and compute residuals
against the original matrix outside timed regions. Both reject nonfinite or
greater-than-1e-8 relative residuals. New runs and matrices live under the
ignored `dev/` directory. The published measurement logs and environment are
checked in under [results/warp9_20261116](results/warp9_20261116).

The explicit build flags prevent an activated Conda environment from replacing
the C++20/AVX2 build with its generic compiler flags. On warp9, MKL is installed
under `$HOME/miniforge3`.

For repeated experiments, create a native CSC cache once. Both benchmark
drivers accept its `.csc` filename directly:

```sh
g++ -O3 -march=native -DNDEBUG -std=c++20 -Isrc \
    benchmarks/cache_matrix.cpp build/native-omp/matrix.o \
    build/native-omp/memory.o -o dev/cache_matrix
dev/cache_matrix dev/CANTILEVER_1M_A.mtx dev/matrix.csc
dev/block_sweep dev/matrix.csc 4 4 off 32,64
```

Cache loading checks dimensions, offsets, sorted lower-triangle indices and
finite values. Cache validation is outside timing. `off` disables serial phase
profiling, and the final argument restricts the tested batch widths.

`order_probe.cpp` reports the ordering, permutation and symbolic times, exact
factor fill, maximum front size and predicted arithmetic work. Its exported
permutation can be selected with `MF_BENCH_ORDER=order.bin` in `block_sweep`.
This isolates ordering quality from numerical kernel changes. Include the
producer's ordering time separately in any end-to-end comparison.

The optional fourth argument to `mkl_reference` exports Pardiso's permutation
after phase 11, using the documented `iparm[4]=2` option. See the
[Intel Pardiso reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso.html).

After building the drivers and creating the cache above, reproduce the paired
batch comparison and MKL reference from a fresh checkout on Linux (Python 3.11+):

```sh
python benchmarks/confirm_warp9.py . --threads 16 --repeat 6 \
    --native dev/block_sweep --mkl dev/mkl_reference \
    --matrix dev/matrix.csc --output dev/confirmation_t16
```

The runner uses the first 16 CPUs in its process affinity unless `--cpus`
specifies an explicit comma-separated list. It records CPU bindings, runtime
settings, the input cache SHA256, raw trials and medians. Runs are sequential;
the first trial of each configuration is excluded from the summary. An optional
`results/cpus.json` can retain a previously sampled CPU selection. The root and
path options also support the original isolated warp9 experiment layout.
