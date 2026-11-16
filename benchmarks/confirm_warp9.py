"""Confirm batch-width performance against the original solver and MKL on warp9."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def records(path):
    return [dict(token.split("=", 1) for token in line.split() if "=" in token)
            for line in path.read_text().splitlines() if line.startswith("trial=")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--cpus", help="Comma-separated CPU indices; defaults to the saved selection or process affinity")
    parser.add_argument("--native", type=Path, help="Path to block_sweep, relative to root or absolute")
    parser.add_argument("--mkl", type=Path, help="Path to mkl_reference, relative to root or absolute")
    parser.add_argument("--matrix", type=Path, help="Path to the input CSC cache, relative to root or absolute")
    parser.add_argument("--output", type=Path, help="Output directory, relative to root or absolute")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.threads < 1 or args.repeat < 3:
        parser.error("Need positive threads and at least three trials")
    if args.cpus:
        try:
            cpus = [int(cpu) for cpu in args.cpus.split(",")]
        except ValueError:
            parser.error("CPU indices must be comma-separated integers")
    elif (root / "results/cpus.json").is_file():
        cpus = json.loads((root / "results/cpus.json").read_text())["cpus"][:args.threads]
    else:
        cpus = sorted(os.sched_getaffinity(0))[:args.threads]
    if len(cpus) != args.threads or len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        parser.error("Need selected CPUs for every thread and at least three trials")
    output = root / (args.output or f"results/confirmation_t{args.threads}")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="ACTIVE", OMP_PROC_BIND="close",
               OMP_PLACES="cores", OMP_THREAD_LIMIT=str(args.threads),
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
               MKL_DYNAMIC="FALSE", MKL_NUM_THREADS=str(args.threads))
    matrix = root / (args.matrix or "model/matrix.csc")
    prefix = ["taskset", "-c", ",".join(map(str, cpus))]
    metadata = {"hostname": platform.node(), "cpus": cpus, "threads": args.threads,
                "repeat": args.repeat, "warmups_per_configuration": 1,
                "load_before": os.getloadavg(), "commands": [], "environment": {
                    name: env[name] for name in env if name.startswith(("OMP_", "MKL_", "OPENBLAS_"))}}
    commands = {
        "native": prefix + [str(root / (args.native or "baseline/block_sweep")), str(matrix),
                            str(args.threads), str(args.repeat), "off", "32,128"],
        "mkl": prefix + [str(root / (args.mkl or "mkl_reference")), str(matrix),
                         str(args.threads), str(args.repeat)],
    }
    for name, command in commands.items():
        metadata["commands"].append(command)
        print("Running", name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print((output / f"{name}.log").read_text(), flush=True)
    rows = []
    for name in ("native", "mkl"):
        data = records(output / f"{name}.log")
        groups = (32, 128) if name == "native" else (None,)
        for width in groups:
            selected = [row for row in data if int(row["trial"]) > 1 and
                        (width is None or int(row["update_block"]) == width)]
            if len(selected) != args.repeat - 1:
                raise RuntimeError("Missing benchmark samples")
            if name == "native" and any(int(row["threads_used"]) != args.threads for row in selected):
                raise RuntimeError("Native solver did not use the requested thread count")
            factor = [float(row["factor_s"]) for row in selected]
            solve = [float(row["solve_s"]) for row in selected]
            rows.append({"solver": name, "block": width, "threads": args.threads,
                         "samples": len(selected), "factor_median_s": statistics.median(factor),
                         "factor_min_s": min(factor), "factor_max_s": max(factor),
                         "solve_median_s": statistics.median(solve),
                         "numeric_median_s": statistics.median(a+b for a, b in zip(factor, solve)),
                         "residual_max": max(float(row["relative_residual"]) for row in selected),
                         "solution_error_max": max(float(row["relative_solution_error"]) for row in selected)})
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with matrix.open("rb") as file:
        metadata["matrix_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    metadata["load_after"] = os.getloadavg()
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    (output / "summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
