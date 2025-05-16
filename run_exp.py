#!/usr/bin/env python3
"""run_experiments.py
===================
Run the five benchmark commands for **every seed requested**, show live
progress (seed + map), collect each *Total reward*, print per‑seed totals and
aggregate statistics.

Examples
--------
```bash
# Seeds 0–9
python run_experiments.py --seed-range 0 9

# Specific seeds
python run_experiments.py --seeds 1,2,3,4,5

# Just show what would run
python run_experiments.py --dry-run
```

Requirements
------------
* Run from the project root (contains *main.py*).
* `main.py` must print a line containing **"Total reward:"** followed by a
  number.
"""

from __future__ import annotations

import argparse
import subprocess
import statistics
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# Each tuple: (map_file, num_agents, n_packages)
EXPERIMENTS: List[Tuple[str, int, int]] = [
    ("map1.txt", 5, 100),
    ("map2.txt", 5, 100),
    ("map3.txt", 5, 500),
    ("map4.txt", 10, 500),
    ("map5.txt", 10, 1000),
]

MAIN_CMD = [sys.executable, "main.py"]
REWARD_RE = re.compile(r"Total reward\s*:\s*([-+]?\d+)")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi‑seed benchmark suite.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", help="Comma‑separated list of seeds, e.g. 0,4,7")
    g.add_argument(
        "--seed-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Run seeds START…END inclusive",
    )
    p.add_argument("--max_time_steps", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def seeds_from_cli(ns: argparse.Namespace) -> List[int]:
    if ns.seeds is not None:
        return [int(s) for s in ns.seeds.split(",") if s.strip()]
    start, end = ns.seed_range
    if start > end:
        start, end = end, start
    return list(range(start, end + 1))


def run_command(cmd: List[str], verbose: int) -> str:
    if verbose:
        print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return proc.stdout + proc.stderr


# ──────────────────────────────────────────────────────────────────────────────
# Core logic


def main() -> None:
    ns = parse_args()
    seeds = seeds_from_cli(ns)

    total_runs = len(seeds) * len(EXPERIMENTS)
    run_idx = 0

    rewards: Dict[int, List[int]] = {seed: [] for seed in seeds}

    for seed in seeds:
        for map_file, num_agents, n_packages in EXPERIMENTS:
            run_idx += 1
            label = f"Seed {seed} » {map_file} ({run_idx}/{total_runs})"
            print(label.ljust(40, "·"), end=" ", flush=True)

            cmd = (
                MAIN_CMD
                + [
                    "--seed",
                    str(seed),
                    "--max_time_steps",
                    str(ns.max_time_steps),
                    "--map",
                    map_file,
                    "--num_agents",
                    str(num_agents),
                    "--n_packages",
                    str(n_packages),
                ]
            )

            if ns.dry_run:
                print("DRY‑RUN")
                continue

            output = run_command(cmd, ns.verbose)
            match = REWARD_RE.search(output)
            if not match:
                print("❌  No reward found; see output above.")
                # dump full output for debugging
                if ns.verbose == 0:
                    print(output)
                sys.exit(1)
            reward = int(match.group(1))
            rewards[seed].append(reward)
            print(f"✔ {reward}")

    if ns.dry_run:
        return

    # ───────────────── aggregated results ────────────────
    print("\nPer‑seed totals")
    print("=" * 15)
    per_seed_totals = []
    for seed in seeds:
        total = sum(rewards[seed])
        per_seed_totals.append(total)
        parts = ", ".join(f"{r:+d}" for r in rewards[seed])
        print(f"Seed {seed:>3}: {total:+6d}  ← [{parts}]")

    print("\nAggregate statistics across seeds")
    print("=" * 34)
    mean = statistics.mean(per_seed_totals)
    median = statistics.median(per_seed_totals)
    stdev = statistics.stdev(per_seed_totals) if len(per_seed_totals) >= 2 else 0.0
    print(f"Mean   : {mean:+.2f}")
    print(f"Median : {median:+.2f}")
    print(f"Min    : {min(per_seed_totals):+d}")
    print(f"Max    : {max(per_seed_totals):+d}")
    print(f"Stdev  : {stdev:.2f}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
