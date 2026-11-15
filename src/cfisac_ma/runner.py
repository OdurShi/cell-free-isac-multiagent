"""Command-line entry point of the experiment scripts: run, save JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from cfisac_ma import experiments as ex
from cfisac_ma import experiments_op as exo

FILES = {
    "e1": "e1_structure",
    "e2": "e2_small_instances",
    "e3": "e3_full_problem",
    "e4": "e4_tradeoff",
    "e5": "e5_adaptation",
    "e6": "e6_maxmin",
    "e7": "e7_channel_model",
    "e8": "e8_blockage",
    "e9": "e9_mode_equilibria",
    "e10": "e10_online_tracking",
    "e11": "e11_locality",
    "e12": "e12_failures",
    "tuning_operation": "tuning_operation",
}
#: tables of each result that are also written as CSV
CSV_TABLES = {
    "e1": ("summary", "extremes", "violations_by_base_size"),
    "e2": ("summary", "minima"),
    "e3": ("summary",),
    "e4": ("summary",),
    "e5": ("summary",),
    "e6": ("summary",),
    "e7": ("summary",),
    "e8": ("summary", "structure", "extremes"),
    "e9": ("summary", "minima", "cost_sweep"),
    "e10": ("summary", "differences", "crossover"),
    "e11": ("summary",),
    "e12": ("summary",),
    "tuning_operation": (),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir(settings: ex.Settings, out: str | None = None) -> Path:
    """``results/`` for the final configuration, ``results/<label>/`` for the others."""
    if out is not None:
        return Path(out)
    base = repo_root() / "results"
    return base if settings.label == ex.FINAL_PASS else base / settings.label


def operation_results_dir(settings: exo.OpSettings, out: str | None = None) -> Path:
    """``results/`` for the final configuration of E8-E12, ``results/operation_<label>/``
    for the others (``results/pass1/`` holds the first pass of E1-E7 only)."""
    if out is not None:
        return Path(out)
    base = repo_root() / "results"
    return base if settings.label == exo.FINAL_PASS else base / f"operation_{settings.label}"


def write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_result(name: str, result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILES[name]}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1, allow_nan=True)
        f.write("\n")
    for table in CSV_TABLES[name]:
        if result.get(table):
            write_csv(out_dir / f"{FILES[name]}_{table}.csv", result[table])
    return path


def load_result(name: str, out_dir: Path | None = None) -> dict:
    out_dir = out_dir if out_dir is not None else repo_root() / "results"
    with open(out_dir / f"{FILES[name]}.json") as f:
        return json.load(f)


def run(name: str, settings: ex.Settings, jobs: int | None = None, out: str | None = None) -> dict:
    t0 = time.time()
    result = ex.EXPERIMENTS[name](settings, jobs)
    path = save_result(name, result, results_dir(settings, out))
    print(f"{name}: {time.time() - t0:.1f} s -> {path}")
    return result


def run_operation(
    name: str, settings: exo.OpSettings, jobs: int | None = None, out: str | None = None
) -> dict:
    """Run one of E8-E12 (or ``tuning_operation``) and save it like :func:`run`."""
    t0 = time.time()
    fn = exo.run_tuning if name == "tuning_operation" else exo.EXPERIMENTS[name]
    result = fn(settings, jobs)
    path = save_result(name, result, operation_results_dir(settings, out))
    print(f"{name}: {time.time() - t0:.1f} s -> {path}")
    return result


def parse_settings(args: argparse.Namespace) -> ex.Settings:
    settings = ex.PASSES[args.settings]
    return ex.quick(settings) if args.quick else settings


def parse_operation_settings(args: argparse.Namespace) -> exo.OpSettings:
    settings = exo.PASSES[args.settings]
    return exo.quick(settings) if args.quick else settings


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings",
        default=ex.FINAL_PASS,
        choices=sorted(ex.PASSES),
        help="configuration (default: the final one)",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="worker processes (default: up to 10)"
    )
    parser.add_argument("--quick", action="store_true", help="reduced sizes, for smoke tests")
    parser.add_argument("--out", default=None, help="output directory (default: results/)")


def main(name: str) -> None:
    if name in exo.EXPERIMENTS or name == "tuning_operation":
        return main_operation(name)
    parser = argparse.ArgumentParser(description=ex.EXPERIMENTS[name].__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    settings = parse_settings(args)
    out = args.out
    if out is None and args.quick:
        out = str(repo_root() / "results" / settings.label)
    run(name, settings, args.jobs, out)


def main_operation(name: str) -> None:
    fn = exo.run_tuning if name == "tuning_operation" else exo.EXPERIMENTS[name]
    parser = argparse.ArgumentParser(description=fn.__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    run_operation(name, parse_operation_settings(args), args.jobs, args.out)
