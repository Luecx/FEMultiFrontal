# Experimental tile-major frontal layout (LDLT)

The production multifrontal LDLT path stores the symmetric frontal matrix
as a packed row-major lower triangle:

```text
C(row,col) = values[row*(row+1)/2+col], row>=col
```

The experimental `--ldlt-tiled-front` switch stores **large fronts**
(physical width >=96) as lower-triangular arrays of row-major **32x32
tiles**. It stores full diagonal tiles but uses only their lower triangle.
Small fronts and compact outgoing contributions stay row-packed.

The physical offset is

```text
TR = physical_row / 32
TC = physical_col / 32
offset = ((TR*(TR+1)/2+TC)*1024 + (physical_row%32)*32 + physical_col%32)
```

where physical indices are `origin + logical_index`, and row >= col.
This convention is invariant under inherited contribution ownership transfers.

The main goal is to keep Schur output tiles adjacent in memory across
successive panel flushes. No output C repacking is required at each flush;
a new 4x8 AVX2/FMA microkernel updates the resident tiles directly.
The left/right rank-major panels are packed once per flush into physical
tile coordinates, with holes from delayed pivots excluded from the target.

A parent can accept row-packed or tiled child contributions. A contiguous
tail may inherit the original tiled buffer if the existing ownership
transfer conditions are met. Otherwise, tile-major contributions are
compacted into canonical row-packed triangles. Contiguous active suffixes
copy each physical tile-row segment in one bulk operation. Tiling is
**opt-in** because its overall effectiveness depends on the relative
costs of faster/lower-locality Schur, additional panel packing, pad
allocation and parent extend-add/copy conversions.

## Build, correctness and paired Warp9 benchmark

```bash
git fetch origin
git switch perf/ldlt-tile-packed-front
git pull --ff-only
make -B -j8 WITH_OPENMP=0 all check \
  CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic"

taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/haken.mtx.zst \
  --binary build/native/mfsolve --threads 1 --ordering nd \
  --warmups 2 --rounds 8 --tiled-front-ablation
taskset -c 90 python3 benchmarks/compare_ldlt_fastpath.py \
  benchmark/indefinite/connector_2_quad.mtx.zst \
  --binary build/native/mfsolve --threads 1 --ordering nd \
  --warmups 2 --rounds 8 --tiled-front-ablation

zstd -dc benchmark/indefinite/haken.mtx.zst >/tmp/haken-tiled.mtx
FEMF_LDLT_PROFILE=1 taskset -c 90 ./build/native/mfsolve \
  /tmp/haken-tiled.mtx --matrix-type symmetric --ordering nd \
  --threads 1 --ldlt-tiled-front --repeat 3
```

Check `tiled_fronts`, `tiled_flushes`, `schur_s`,
`extend_add_s`, `contribution_copy_s`, `front_zero_s` and
`factor_s`. The first two fields confirm the experiment was used;
phase time improvements alone do not imply a faster total factorization.

The unit test compares all meaningful entries against an independent
scalar Schur reference for multiple dimensions, inherited physical
origins, active-index holes and signs. It also checks untouched
upper/padding regions. The end-to-end regression compares the original
row-packed and experimental tile-major factorizations on n>=96 symmetric
indefinite, singular and ill-conditioned matrices, including 2x2 and
delayed pivots. Residual, rank and inertia must remain valid.

**Do not merge merely because correctness tests pass.** The new
layout has additional costs and should only become the default after
a reproducible factorization speedup on `haken` and no material
regression on `connector_2_quad` or other representative matrices.
