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
            "--pivot-tolerance", str(args.pivot_tolerance),
            "--rank-tolerance", str(args.rank_tolerance),
        ]
        if args.cpu is not None:
            command = ["taskset", "-c", str(args.cpu)] + command

        variants = (
            {"previous": ("--ldlt-no-pivot-cache",),
             "candidate": ()}
            if args.ablations else {"legacy": LEGACY, "candidate": ()}
        )
        labels = list(variants)

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
        baseline = "previous" if args.ablations else "legacy"
        print(f"speedup={medians[baseline]/medians['candidate']:.5f}x rank_and_inertia={reference}")

if __name__ == "__main__":
    main()
