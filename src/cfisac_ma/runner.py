"""Command-line entry point of the experiment scripts: run, save JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from cfisac_ma import experiments as ex

FILES = {
    "e1": "e1_structure",
    "e2": "e2_small_instances",
    "e3": "e3_full_problem",
    "e4": "e4_tradeoff",
    "e5": "e5_adaptation",
    "e6": "e6_maxmin",
    "e7": "e7_channel_model",
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
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir(settings: ex.Settings, out: str | None = None) -> Path:
    """``results/`` for the final configuration, ``results/<label>/`` for the others."""
    if out is not None:
        return Path(out)
    base = repo_root() / "results"
    return base if settings.label == ex.FINAL_PASS else base / settings.label


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


def parse_settings(args: argparse.Namespace) -> ex.Settings:
    settings = ex.PASSES[args.settings]
    return ex.quick(settings) if args.quick else settings


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
    parser = argparse.ArgumentParser(description=ex.EXPERIMENTS[name].__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    settings = parse_settings(args)
    out = args.out
    if out is None and args.quick:
        out = str(repo_root() / "results" / settings.label)
    run(name, settings, args.jobs, out)
