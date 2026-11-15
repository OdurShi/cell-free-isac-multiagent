"""Reproduce every result file, table and figure of the repository with one command.

Usage: python scripts/reproduce_all.py [--suite placement|operation|all] [--jobs N] [--quick]
                                       [--skip-tuning] [--skip-figures]
                                       [--settings pass1|pass2] [--out DIR]

Two suites, both on the shipped data in ``data/``:

``placement`` (E1-E7)
    1. hyper-parameter grids of the stochastic methods on separate tuning scenarios
       (``results/tuning.json``; the experiments use the values stored in the settings, and
       the script reports whether they are the selected grid points),
    2. experiments E1-E7 (``results/e*.json`` and CSV summaries),
    3. figures (``figures/e1_*.png`` ... ``figures/e7_*.png``, ``figures/teaser.png``),
    4. tables (``results/tables.md``).

    Times are written to ``results/runtime.json``.

``operation`` (E8-E12)
    1. calibration of the costs and grids of the stochastic methods on the tuning scenarios
       (``results/tuning_operation.json``),
    2. experiments E8-E12,
    3. figures (``figures/e8_*.png`` ... ``figures/e12_*.png``,
       ``figures/teaser_operation.png``),
    4. tables (``results/tables.md``, all sections whose result files are present).

    Times are written to ``results/runtime_operation.json``.

``all`` (default) runs both.  ``--settings pass1 --skip-tuning --skip-figures`` regenerates
the first-pass results in ``results/pass1/`` (placement) and ``results/operation_pass1/``
(operation).
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
from cfisac_ma import experiments_op as exo
from cfisac_ma.runner import (
    operation_results_dir,
    parse_operation_settings,
    parse_settings,
    repo_root,
    results_dir,
    run,
    run_operation,
)

HERE = Path(__file__).resolve().parent


def finish(out: Path, suite: str, steps: dict, args, label: str, t_all: float, name: str) -> None:
    """Figures and tables of a suite, then its runtime file."""
    jobs = args.jobs or ex.default_jobs()
    figures = repo_root() / "figures" if out == repo_root() / "results" else out / "figures"
    for step, script, extra in (
        ("figures", "make_figures.py", ["--out", str(figures), "--suite", suite]),
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
    with open(out / name, "w") as f:
        json.dump(
            {
                "settings": label,
                "jobs": jobs,
                "total_s": total,
                "load_average_at_end": [round(v, 2) for v in os.getloadavg()],
                "steps_s": steps,
                "machine": ex.machine_info(),
            },
            f,
            indent=1,
        )
        f.write("\n")
    print(f"{suite}: total {total} s; steps: {steps}")


def placement(args) -> None:
    settings = parse_settings(args)
    out = Path(args.out) if args.out else results_dir(settings)
    if args.quick and args.out is None:
        out = repo_root() / "results" / settings.label
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
    finish(out, "placement", steps, args, settings.label, t_all, "runtime.json")


def operation(args) -> None:
    settings = parse_operation_settings(args)
    out = Path(args.out) if args.out else operation_results_dir(settings)
    steps: dict[str, float] = {}
    t_all = time.time()
    names = list(exo.EXPERIMENTS)
    if not args.skip_tuning:
        names.insert(0, "tuning_operation")
    for name in names:
        t0 = time.time()
        run_operation(name, settings, args.jobs, str(out))
        steps["tuning" if name == "tuning_operation" else name] = round(time.time() - t0, 1)
    finish(out, "operation", steps, args, settings.label, t_all, "runtime_operation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", default="all", choices=("placement", "operation", "all"))
    parser.add_argument("--settings", default=ex.FINAL_PASS, choices=sorted(ex.PASSES))
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="reduced sizes, for smoke tests")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.suite in ("placement", "all"):
        placement(args)
    if args.suite in ("operation", "all"):
        operation(args)


if __name__ == "__main__":
    main()
