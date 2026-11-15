"""Reproduce every result file, table and figure of the repository with one command.

Usage: python scripts/reproduce_all.py [--jobs N] [--quick] [--skip-tuning] [--skip-figures]
                                       [--settings pass1|pass2]

Steps (all on the shipped data in ``data/``):

1. hyper-parameter grids of the stochastic methods on separate tuning scenarios
   (``results/tuning.json``; the experiments use the values stored in the settings, and the
   script reports whether they are the selected grid points),
2. experiments E1-E7 (``results/e*.json`` and CSV summaries),
3. figures (``figures/*.png``),
4. tables for the README (``results/tables.md``).

The total wall-clock time and the time of every step are written to
``results/runtime.json``.  ``--settings pass1 --skip-tuning --skip-figures`` regenerates the
first-pass results in ``results/pass1/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cfisac_ma import experiments as ex
from cfisac_ma.runner import parse_settings, repo_root, results_dir, run

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--settings", default=ex.FINAL_PASS, choices=sorted(ex.PASSES))
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="reduced sizes, for smoke tests")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    settings = parse_settings(args)
    out = Path(args.out) if args.out else results_dir(settings)
    if args.quick and args.out is None:
        out = repo_root() / "results" / settings.label
    figures = repo_root() / "figures" if out == repo_root() / "results" else out / "figures"
    jobs = ["--jobs", str(args.jobs)] if args.jobs else []

    steps: dict[str, float] = {}
    t_all = time.time()
    if not args.skip_tuning and not args.quick:
        t0 = time.time()
        subprocess.run(
            [sys.executable, str(HERE / "tune_baselines.py"), "--out", str(out), *jobs], check=True
        )
        steps["tuning"] = round(time.time() - t0, 1)
    for name in ex.EXPERIMENTS:
        t0 = time.time()
        run(name, settings, args.jobs, str(out))
        steps[name] = round(time.time() - t0, 1)
    for step, script, extra in (
        ("figures", "make_figures.py", ["--out", str(figures)]),
        ("tables", "make_tables.py", []),
    ):
        if step == "figures" and args.skip_figures:
            continue
        t0 = time.time()
        subprocess.run(
            [sys.executable, str(HERE / script), "--results", str(out), *extra], check=True
        )
        steps[step] = round(time.time() - t0, 1)
    total = round(time.time() - t_all, 1)
    with open(out / "runtime.json", "w") as f:
        json.dump(
            {
                "settings": settings.label,
                "jobs": args.jobs or ex.default_jobs(),
                "total_s": total,
                "load_average_at_end": [round(v, 2) for v in os.getloadavg()],
                "steps_s": steps,
                "machine": ex.machine_info(),
            },
            f,
            indent=1,
        )
        f.write("\n")
    print(f"total: {total} s; steps: {steps}")


if __name__ == "__main__":
    main()
