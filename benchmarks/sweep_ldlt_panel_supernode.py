#!/usr/bin/env python3
"""Interleaved LDLT panel-size / supernode-size factorial benchmark.

The existing --ldlt-block-size sets both the panel rank and the supernode
amalgamation width. This tool evaluates them independently on identical
symmetric MatrixMarket input, including zstd-compressed matrices. One run
is one complete refactorization (analysis outside the timed factor interval).
"""
import argparse
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

FIELD = re.compile(r"\b([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)")
CHECK_KEYS = (
    "rank_estimate", "nullity_estimate",
    "inertia_positive", "inertia_negative", "inertia_zero",
)

def parse_pairs(arg):
    out = []
    for part in arg.split(","):
        values = part.split(":")
        if len(values) != 2:
            raise argparse.ArgumentTypeError(
                "Expected --pairs PANEL:SUPERNODE,PANEL:SUPERNODE,..."
            )
        try:
            p, s = (int(x) for x in values)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Panel/supernode must be integers") from exc
        if not (2 <= p <= 256 and 1 <= s <= 256):
            raise argparse.ArgumentTypeError("Panel must be in [2,256], supernode in [1,256]")
        pair = (p, s)
        if pair in out:
            raise argparse.ArgumentTypeError("Repeated panel/supernode pair")
        out.append(pair)
    if not out:
        raise argparse.ArgumentTypeError("At least one pair required")
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/native/mfsolve"))
    parser.add_argument("--pairs", type=parse_pairs,
                        default=parse_pairs("32:32,32:64,64:32,64:64"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=None,
                        help="Optional Linux CPU affinity; omit when not pinning")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ordering", choices=("nd", "natural"), default="nd")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("Invalid rounds/warmups/threads")
    binary = str(args.binary.resolve())
    env = dict(os.environ)
    env.pop("FEMF_LDLT_PROFILE", None)
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")

    with tempfile.TemporaryDirectory(prefix="femf-panel-group-") as folder:
        path = args.matrix.resolve()
        if str(path).endswith(".zst"):
            destination = Path(folder) / path.name.removesuffix(".zst")
            with destination.open("wb") as output:
                subprocess.run(["zstd", "-d", "-q", "-c", str(path)],
                               stdout=output, check=True)
            path = destination
        data = {pair: [] for pair in args.pairs}
        signature = None
        print(f"matrix={args.matrix} pairs={args.pairs} rounds={args.rounds} "
              f"warmups={args.warmups} threads={args.threads} cpu={args.cpu}",
              flush=True)
        for round_idx in range(-args.warmups, args.rounds):
            # Rotate measurement order to reduce time-dependent bias.
            start = (round_idx + args.warmups) % len(args.pairs)
            order = args.pairs[start:] + args.pairs[:start]
            if ((round_idx + args.warmups) // len(args.pairs)) % 2:
                order.reverse()
            for panel, supernode in order:
                cmd = [
                    binary, str(path), "--matrix-type", "symmetric",
                    "--ordering", args.ordering, "--threads", str(args.threads),
                    "--ldlt-block-size", str(panel),
                    "--ldlt-supernode-size", str(supernode),
                ]
                if args.cpu is not None:
                    cmd = ["taskset", "-c", str(args.cpu)] + cmd
                proc = subprocess.run(cmd, text=True, capture_output=True,
                                      env=env, check=True)
                trial = next((line for line in proc.stdout.splitlines()
                              if line.startswith("trial=1 ")), None)
                if not trial:
                    raise RuntimeError(f"Missing trial data: {proc.stdout[-4000:]}")
                values = dict(FIELD.findall(trial))
                factor = float(values["factor_s"])
                residual = float(values["relative_residual"])
                sig = tuple(values[k] for k in CHECK_KEYS)
                if not (0 < factor and 0 <= residual <= 1e-8):
                    raise RuntimeError(f"Invalid result: {trial}")
                if signature is None:
                    signature = sig
                elif sig != signature:
                    raise RuntimeError(
                        f"Different numerical rank/inertia: {sig} != {signature}, "
                        f"panel={panel} supernode={supernode}"
                    )
                if round_idx >= 0:
                    data[(panel, supernode)].append(factor)
                    print(f"round={round_idx+1} panel={panel} supernode={supernode}"
                          f" factor_s={factor:.9f} residual={residual:.3g}",
                          flush=True)
        print("rank_inertia=" + str(dict(zip(CHECK_KEYS, signature))), flush=True)
        results = []
        for panel, supernode in args.pairs:
            times = data[(panel, supernode)]
            med = statistics.median(times)
            mad = statistics.median(abs(x-med) for x in times)
            results.append((med, panel, supernode))
            print(f"panel={panel} supernode={supernode} "
                  f"factor_median_s={med:.9f} factor_MAD_s={mad:.9f}", flush=True)
        best = min(results)
        print(f"best_panel={best[1]} best_supernode={best[2]} "
              f"best_factor_median_s={best[0]:.9f}", flush=True)

if __name__ == "__main__":
    main()
