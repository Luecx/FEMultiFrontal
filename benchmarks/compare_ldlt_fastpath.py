#!/usr/bin/env python3
"""Paired LDLT A/B benchmark within one binary: legacy vs new front fast paths.

Example:
  python3 benchmarks/compare_ldlt_fastpath.py /data/haken.mtx.zst \
      --binary build/native-omp/mfsolve --threads 4 --rounds 6 --cpu 90

Decompression, MatrixMarket loading, ND ordering and symbolic analysis are
excluded from factor_s by mfsolve. All runs use identical input, compiler,
ordering, thread count, numerical thresholds and executable.
"""
import argparse
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

LEGACY = (
    "--ldlt-legacy-pivot-order",
    "--ldlt-no-tiled-schur",
    "--ldlt-no-pivot-cache",
    "--ldlt-no-relaxed-supernodes",
)
METRIC = re.compile(r"\b(\w+)=([^\s]+)")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="MatrixMarket .mtx or .mtx.zst")
    parser.add_argument("--binary", default="build/native/mfsolve")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--ablations", action="store_true", help="Compare PR23 baseline with the pivot-cache-only candidate")
    parser.add_argument("--indexed-ablation", action="store_true",
                        help="Compare current production Schur kernel with new packed-indexed kernel")
    parser.add_argument("--pivot-major-cache-ablation", action="store_true",
                        help="Compare scalar per-entry cache rebuild with pivot-major SIMD refresh")
    parser.add_argument("--simd-cache-apply-ablation", action="store_true",
                        help="Compare SIMD vs scalar cache row updates on top of pivot-major cache refresh")
    parser.add_argument("--signed-batch-simd-columns-ablation", action="store_true",
                        help="Compare scalar and packed SIMD pivot-column evaluation")
    parser.add_argument("--signed-batch-register-blocking-ablation", action="store_true",
                        help="Compare 1x4 and 4x4 signed batch AVX2 kernels")
    parser.add_argument("--signed-batch-simd-rectangle-ablation", action="store_true",
                        help="Compare signed batch rectangular updates with and without explicit AVX2/FMA")
    parser.add_argument("--signed-batch-skip-diagonal-ablation", action="store_true",
                        help="Compare signed batches with and without redundant diagonal cache updates")
    parser.add_argument("--signed-batch-ablation", action="store_true",
                        help="Compare merged-main cache path with signed threshold batch panels")
    parser.add_argument("--signed-batch-width", type=int, default=16)
    parser.add_argument("--segmented-extend-add-ablation", action="store_true",
                        help="Compare segmented vs scalar child-to-parent extend-add on top of both cache optimizations")
    parser.add_argument("--contribution-copy-ablation", action="store_true",
                        help="Compare exact original contribution compaction with optimized packed row copies")
    parser.add_argument("--front-prepare-ablation", action="store_true",
                        help="Compare original concatenation/sort versus sorted-union front construction")
    parser.add_argument("--ordering", choices=("nd", "natural"), default="nd")
    parser.add_argument("--cpu", type=int, help="Optional Linux taskset CPU; leave empty for multi-thread runs")
    parser.add_argument("--pivot-tolerance", type=float, default=0.1)
    parser.add_argument("--rank-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.threads < 1 or args.rounds < 1 or args.warmups < 0:
        parser.error("threads/rounds must be positive and warmups nonnegative")
    with tempfile.TemporaryDirectory(prefix="femf-ldlt-ab-") as folder:
        path = args.matrix.resolve()
        if str(path).endswith(".zst"):
            uncompressed = Path(folder) / path.name.removesuffix(".zst")
            with uncompressed.open("wb") as output:
                subprocess.run(["zstd", "-q", "-d", "-c", str(path)], stdout=output, check=True)
            path = uncompressed
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.threads)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env.pop("FEMF_LDLT_PROFILE", None)
        command = [
            str(Path(args.binary).resolve()), str(path),
            "--matrix-type", "symmetric",
            "--ordering", args.ordering,
            "--threads", str(args.threads),
            "--ldlt-block-size", str(args.block_size),
            "--ldlt-signed-batch-width", str(args.signed_batch_width),
            "--pivot-tolerance", str(args.pivot_tolerance),
            "--rank-tolerance", str(args.rank_tolerance),
        ]
        if args.cpu is not None:
            command = ["taskset", "-c", str(args.cpu)] + command

        if sum((args.ablations, args.indexed_ablation, args.contribution_copy_ablation,
                args.front_prepare_ablation, args.pivot_major_cache_ablation,
                args.simd_cache_apply_ablation,
                args.segmented_extend_add_ablation, args.signed_batch_ablation,
                args.signed_batch_skip_diagonal_ablation, args.signed_batch_simd_rectangle_ablation, args.signed_batch_register_blocking_ablation, args.signed_batch_simd_columns_ablation)) > 1:
            parser.error("Ablation switches are mutually exclusive")
        variants = (
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache", "--ldlt-signed-batch-simd-rectangle", "--ldlt-signed-batch-register-blocking"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache", "--ldlt-signed-batch-simd-rectangle", "--ldlt-signed-batch-register-blocking", "--ldlt-signed-batch-simd-columns")}
            if args.signed_batch_simd_columns_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache", "--ldlt-signed-batch-simd-rectangle"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache", "--ldlt-signed-batch-simd-rectangle", "--ldlt-signed-batch-register-blocking")}
            if args.signed_batch_register_blocking_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache", "--ldlt-signed-batch-simd-rectangle")}
            if args.signed_batch_simd_rectangle_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels", "--ldlt-signed-batch-skip-diagonal-cache")}
            if args.signed_batch_skip_diagonal_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-signed-batch-panels")}
            if args.signed_batch_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply"),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply", "--ldlt-segmented-extend-add")}
            if args.segmented_extend_add_ablation else
            {"previous": ("--ldlt-pivot-major-cache-refresh",),
             "candidate": ("--ldlt-pivot-major-cache-refresh", "--ldlt-simd-cache-apply")}
            if args.simd_cache_apply_ablation else
            {"previous": (), "candidate": ("--ldlt-pivot-major-cache-refresh",)}
            if args.pivot_major_cache_ablation else
            {"previous": ("--ldlt-no-fast-front-prepare",), "candidate": ()}
            if args.front_prepare_ablation else
            {"previous": ("--ldlt-no-fast-contribution-copy",), "candidate": ()}
            if args.contribution_copy_ablation else
            {"previous": ("--ldlt-no-indexed-schur",), "candidate": ()}
            if args.indexed_ablation else
            {"previous": ("--ldlt-no-pivot-cache",), "candidate": ()}
            if args.ablations else
            {"legacy": LEGACY, "candidate": ()}
        )
        labels = list(variants)

        batch_diagnostics = {}

        def measure(label):
            flags = variants[label]
            completed = subprocess.run(
                command + list(flags), check=True, capture_output=True,
                text=True, env=env,
            )
            trial = next((line for line in completed.stdout.splitlines()
                         if line.startswith("trial=1 ")), None)
            if trial is None:
                raise RuntimeError("Missing trial metrics: "+completed.stdout[-2000:])
            data = dict(METRIC.findall(trial))
            if (args.signed_batch_skip_diagonal_ablation or args.signed_batch_simd_rectangle_ablation
                    or args.signed_batch_register_blocking_ablation or args.signed_batch_simd_columns_ablation):
                summary = next((line for line in completed.stdout.splitlines()
                                if "signed_batch_skipped_diagonal_updates=" in line), "")
                metrics = dict(METRIC.findall(summary))
                diagnostics = {key: metrics[key] for key in (
                    "signed_batch_attempts", "signed_batch_pivots",
                    "signed_batch_skipped_diagonal_updates", "signed_batch_simd_entries",
                    "signed_batch_register_block_entries", "signed_batch_simd_column_entries") if key in metrics}
                if diagnostics and label not in batch_diagnostics:
                    batch_diagnostics[label] = diagnostics
            factor = float(data["factor_s"])
            residual = float(data["relative_residual"])
            if not (factor > 0 and residual <= 1e-8):
                raise RuntimeError("Invalid factor time or residual: "+trial)
            return factor, {name: data[name] for name in (
                "rank_estimate", "nullity_estimate",
                "inertia_positive", "inertia_negative", "inertia_zero"
            ) if name in data}

        print(f"matrix={args.matrix} threads={args.threads} block_size={args.block_size} ordering={args.ordering}")
        print(f"rounds={args.rounds} warmups={args.warmups} executable={args.binary}", flush=True)
        reference = None
        for i in range(args.warmups):
            for label in labels:
                _, state = measure(label)
                if reference is None:
                    reference = state
                if state != reference:
                    raise RuntimeError(f"Rank/inertia mismatch in warmup: {state} != {reference}")
        observations = {label: [] for label in labels}
        for i in range(args.rounds):
            for label in (labels if i%2 == 0 else list(reversed(labels))):
                seconds, state = measure(label)
                if reference is None:
                    reference = state
                if state != reference:
                    raise RuntimeError(f"Rank/inertia mismatch: {label}: {state} != {reference}")
                observations[label].append(seconds)
                print(f"round={i+1} variant={label} factor_s={seconds:.9f}",flush=True)
        medians = {label: statistics.median(values) for label, values in observations.items()}
        for label, values in observations.items():
            median = medians[label]
            mad = statistics.median(abs(x-median) for x in values)
            print(f"{label}_median_s={median:.9f} {label}_MAD_s={mad:.9f}")
        baseline = "previous" if any((args.ablations, args.indexed_ablation,
                                     args.contribution_copy_ablation,
                                     args.front_prepare_ablation,
                                     args.pivot_major_cache_ablation,
                                     args.simd_cache_apply_ablation,
                                     args.segmented_extend_add_ablation,
                                     args.signed_batch_ablation, args.signed_batch_skip_diagonal_ablation,
                                     args.signed_batch_simd_rectangle_ablation,
                                     args.signed_batch_register_blocking_ablation,
                                     args.signed_batch_simd_columns_ablation)) else "legacy"
        print(f"speedup={medians[baseline]/medians['candidate']:.5f}x rank_and_inertia={reference}")
        for label, diagnostics in batch_diagnostics.items():
            print(f"variant={label} " + " ".join(
                f"{key}={value}" for key, value in diagnostics.items()))

if __name__ == "__main__":
    main()
