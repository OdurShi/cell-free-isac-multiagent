"""Draw all figures of the repository from the result files.

Usage: python scripts/make_figures.py [--results DIR] [--out DIR] [--only NAME ...]
                                      [--suite placement|operation|all]

Reads ``results/*.json`` (written by the ``run_e*.py`` scripts) and the shipped data and
writes PNG files (200 dpi) to ``figures/``.  ``--suite`` restricts the figures to those of
the placement study (E1-E7, ``teaser``) or of the operation study (E8-E12,
``teaser_operation``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cfisac_ma import plotting as pl
from cfisac_ma import plotting_op as plo
from cfisac_ma.runner import load_result, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=repo_root() / "results")
    parser.add_argument("--out", type=Path, default=repo_root() / "figures")
    parser.add_argument("--only", nargs="*", default=None, help="figure names (default: all)")
    parser.add_argument("--suite", default="all", choices=("placement", "operation", "all"))
    args = parser.parse_args()

    def load(name):
        return load_result(name, args.results)

    figures = {
        "teaser": lambda p: pl.figure_teaser(p),
        "e1_structure": lambda p: pl.figure_e1(load("e1"), p),
        "e1_eps": lambda p: pl.figure_e1_eps(load("e1"), p),
        "e2_small_instances": lambda p: pl.figure_e2(load("e2"), p),
        "e3_curves": lambda p: pl.figure_curves(load("e3"), p),
        "e3_final": lambda p: pl.figure_final(load("e3"), p),
        "e3_convergence": lambda p: pl.figure_convergence(load("e3"), p),
        "e3_users": lambda p: pl.figure_users(load("e3"), p),
        "e3_credit": lambda p: pl.figure_credit(load("e3"), p),
        "e4_tradeoff": lambda p: pl.figure_e4(load("e4"), p),
        "e5_adaptation": lambda p: pl.figure_e5(load("e5"), p),
        "e6_maxmin_curves": lambda p: pl.figure_curves(load("e6"), p, ylim=(0.0, 1.02)),
        "e6_maxmin_final": lambda p: pl.figure_final(
            load("e6"), p, panels=("ratio", "min_rate_positive")
        ),
        "e7_channel_model": lambda p: pl.figure_e7(load("e7"), p),
    }
    operation = plo.figures(load)
    if args.suite == "operation":
        figures = operation
    elif args.suite == "all":
        figures.update(operation)
    names = args.only or list(figures)
    for name in names:
        path = figures[name](args.out / f"{name}.png")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
