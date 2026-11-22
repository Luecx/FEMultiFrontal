# Optional METIS ordering comparison

FEMF's native `--ordering nd` remains the default and has no external dependency.
For benchmarking an independent implementation of nested dissection, the
`--ordering metis` option calls METIS_NodeND on the **scalar, undirected
structural graph**. It is available only when built with `WITH_METIS=1`.

The METIS result is translated into FEMF's `order[new] = original` convention.
Both the SPD and rank-aware LDLT/LU drivers accept the same option.

## Install and build (Conda / Miniforge)

```bash
conda install -c conda-forge metis
make -j8 WITH_METIS=1 all check CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"
```

The optional binary is `build/native-metis/mfsolve`; it does not overwrite
`build/native/mfsolve`. Without METIS installed, the normal build works as
before. Runtime errors explain how to enable METIS if requested in a normal
build.

## Evaluate the ordering rather than just runtime

```bash
M=/tmp/connector_2_quad.mtx
EXE=./build/native-metis/mfsolve
$EXE "$M" --ordering nd --analyze-only
$EXE "$M" --ordering metis --analyze-only
```

This SPD-symbolic comparison reports `factor_nnz`, `factor_flops` and
`max_front`. These are structural Cholesky estimates, **not** the actual
flop count for dynamically pivoted rank-aware LDLT.

To compare numerical timings, pass `--matrix-type symmetric` to select LDLT:

```bash
$EXE "$M" --matrix-type symmetric --ordering nd --threads 1 --ldlt-block-size 16 --repeat 12
$EXE "$M" --matrix-type symmetric --ordering metis --threads 1 --ldlt-block-size 16 --repeat 12
```

Check `relative_residual`, `nullity_estimate` and inertia before comparing
factorization times. Measure under consistent CPU load; ordering and symbolic
analysis are excluded from `factor_s`.

METIS currently sees the uncompressed scalar graph. FEMF's native ordering
compresses structurally similar degrees of freedom, so METIS is not assumed to
be faster. Its main role here is as an independent, measured reference.
