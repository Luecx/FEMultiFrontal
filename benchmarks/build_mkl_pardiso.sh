#!/usr/bin/env bash
# Build and optionally run an isolated oneMKL PARDISO reference benchmark.
# Does not change FEMF's normal build.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
roots=()
[[ -n "${MKLROOT:-}" ]] && roots+=("$MKLROOT")
[[ -n "${CONDA_PREFIX:-}" ]] && roots+=("$CONDA_PREFIX")
roots+=("/opt/intel/oneapi/mkl/latest" "/usr/local/intel/oneapi/mkl/latest")

mkl_inc=""
mkl_lib=""
for root in "${roots[@]}"; do
    [[ -f "$root/include/mkl.h" ]] || continue
    for lib in "$root/lib" "$root/lib/intel64"; do
        if [[ -f "$lib/libmkl_rt.so" ]]; then
            mkl_inc="$root/include"
            mkl_lib="$lib"
            break 2
        fi
    done
done

if [[ -z "$mkl_lib" ]]; then
    cat >&2 <<'HELP'
Cannot locate mkl.h and libmkl_rt.so.
Load the installed MKL module or activate its Conda environment.
Then inspect: echo "MKLROOT=$MKLROOT CONDA_PREFIX=$CONDA_PREFIX"
If MKL is in a custom prefix, export MKLROOT=<prefix containing include/ and lib/>.
HELP
    exit 1
fi

exe="$repo/build/native/mkl_pardiso_bench"
mkdir -p "$(dirname "$exe")"
echo "oneMKL include: $mkl_inc"
echo "oneMKL runtime: $mkl_lib"
"${CXX:-g++}" -O3 -march=native -DNDEBUG -std=c++20 \
    -Wall -Wextra -Wpedantic \
    -I"$repo/src" -I"$mkl_inc" \
    "$repo/benchmarks/mkl_pardiso.cpp" "$repo/src/matrix.cpp" \
    -L"$mkl_lib" -Wl,-rpath,"$mkl_lib" \
    -lmkl_rt -lpthread -lm -ldl -o "$exe"

if [[ $# -gt 0 ]]; then
    export LD_LIBRARY_PATH="$mkl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_DYNAMIC=FALSE
    export MKL_DYNAMIC=FALSE
    export MKL_INTERFACE_LAYER=LP64
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    exec "$exe" "$@"
fi
echo "Built: $exe"
