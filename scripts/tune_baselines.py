"""Hyper-parameter grids of the stochastic methods on separate tuning scenarios.

Usage: python scripts/tune_baselines.py [--jobs N] [--out DIR]

The tuning scenarios use the user-draw seeds 1000-1004, which no experiment uses.  Every
configuration is run with the evaluation budget of the experiments (``6 N V``) for N = 8
agents, two repetitions per scenario.  Selection rule: the largest mean final ratio
``best utility / best utility known for the scenario`` (rounded to four decimals); ties are
broken by the anytime score, the mean of the same ratio at 1/8, 1/4, 1/2 and all of the
budget.  The result is written to ``results/tuning.json``; the final settings
(``cfisac_ma.experiments.PASSES``) contain the selected values, and the script reports
whether they agree with the selected grid point.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cfisac_ma import experiments as ex
from cfisac_ma import team
from cfisac_ma.objective import trace_value_at
from cfisac_ma.runner import repo_root
from cfisac_ma.stats import mean_ci

TUNING_SEEDS = (1000, 1001, 1002, 1003, 1004)
N_AGENTS = 8
REPS = 2
CHECKPOINTS = (0.125, 0.25, 0.5, 1.0)
GRIDS = {
    "sa": [
        {"t_start": t0, "t_end": t1, "cooling_fraction": frac}
        for t0, t1, frac in itertools.product((3e-2, 1e-2, 3e-3), (1e-5, 1e-6), (0.1, 0.3, 1.0))
    ]
    + [{"t_start": 100.0, "cooling": 0.99}],
    "ga": [
        {"population": p, "mutation_rate": m}
        for p, m in itertools.product((20, 50, 100), (0.1, 0.3, 0.6))
    ],
    "bandit-team": [
        {"batch": b, "learning_rate": lr}
        for b, lr in itertools.product((16, 32, 64), (1.0, 2.0, 4.0, 8.0, 16.0))
    ],
    "bandit-diff": [
        {"batch": b, "learning_rate": lr}
        for b, lr in itertools.product((4, 8, 16, 32), (1.0, 2.0, 4.0, 8.0, 16.0))
    ],
    "sbr": [
        {"n_random": r, "n_near": k, "patience_rounds": 5}
        for r, k in itertools.product((8, 16, 32), (0, 8, 16))
    ],
}
FIELD = {
    "sa": "sa",
    "ga": "ga",
    "bandit-team": "bandit_team",
    "bandit-diff": "bandit_diff",
    "sbr": "sbr",
}


def _task(args) -> dict:
    """One method (all its grid points) or the reference run on one tuning scenario."""
    seed, method = args
    base = ex.PASSES[ex.FINAL_PASS]
    problem = ex.scenario_problem(base, seed)
    budget = ex.budget_for(base, N_AGENTS, problem.n_sites)
    if method == "reference":
        # multi-start best response with four times the budget
        ref = ex.run_method(
            "br",
            problem,
            N_AGENTS,
            4 * budget,
            np.random.default_rng([92, seed]),
            base,
            team.random_placement(np.random.default_rng([93, seed]), problem.n_sites, N_AGENTS),
        )
        return {"scenario": seed, "reference": ref.best_value, "rows": []}
    checkpoints = np.round(np.array(CHECKPOINTS) * budget).astype(np.int64)
    rows = []
    for j, params in enumerate(GRIDS[method]):
        settings = replace(base, **{FIELD[method]: params})
        for rep in range(REPS):
            init = team.random_placement(
                np.random.default_rng([90, seed, rep]), problem.n_sites, N_AGENTS
            )
            rng = np.random.default_rng([91, seed, ex.METHODS.index(method), rep])
            res = ex.run_method(method, problem, N_AGENTS, budget, rng, settings, init)
            rows.append(
                {
                    "scenario": seed,
                    "method": method,
                    "config": j,
                    "rep": rep,
                    "values": trace_value_at(res.trace, checkpoints).tolist(),
                }
            )
    return {"scenario": seed, "reference": -np.inf, "rows": rows}


def _collect(parts: list[dict]) -> list[dict]:
    """Turn the checkpoint values into ratios to the best utility known per scenario."""
    rows = [r for part in parts for r in part["rows"]]
    for seed in TUNING_SEEDS:
        sel = [r for r in rows if r["scenario"] == seed]
        best = max(
            [p["reference"] for p in parts if p["scenario"] == seed]
            + [r["values"][-1] for r in sel]
        )
        for r in sel:
            ratios = np.nan_to_num(np.array(r.pop("values")) / best, nan=0.0)
            r["final_ratio"] = float(ratios[-1])
            r["anytime_score"] = float(ratios.mean())
    return rows


def _per_scenario(rows: list[dict], method: str, config: int, key: str) -> list[float]:
    return [
        float(
            np.mean(
                [
                    r[key]
                    for r in rows
                    if r["method"] == method and r["config"] == config and r["scenario"] == s
                ]
            )
        )
        for s in TUNING_SEEDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=repo_root() / "results")
    args = parser.parse_args()
    t0 = time.time()
    tasks = [(seed, method) for method in ("reference", *GRIDS) for seed in TUNING_SEEDS]
    rows = _collect(ex.parallel_map(_task, tasks, args.jobs))
    final = ex.PASSES[ex.FINAL_PASS]
    table, selected = [], {}
    for method, grid in GRIDS.items():
        scores = []
        used_params = getattr(final, FIELD[method])
        for j, params in enumerate(grid):
            score = mean_ci(_per_scenario(rows, method, j, "anytime_score"))
            last = mean_ci(_per_scenario(rows, method, j, "final_ratio"))
            scores.append((round(last["mean"], 4), score["mean"]))
            table.append(
                {
                    "method": method,
                    "config": j,
                    "parameters": params,
                    "anytime_score": score["mean"],
                    "anytime_score_lo": score["lo"],
                    "anytime_score_hi": score["hi"],
                    "final_ratio": last["mean"],
                    "final_ratio_lo": last["lo"],
                    "final_ratio_hi": last["hi"],
                    "used_in_final_settings": params == used_params,
                }
            )
        best = max(range(len(grid)), key=lambda j: scores[j])
        selected[method] = {
            "selected_config": grid[best],
            "final_ratio": scores[best][0],
            "anytime_score": scores[best][1],
            "final_settings": used_params,
            "final_settings_are_selected": grid[best] == used_params,
        }
        print(
            f"{method:12s} selected {grid[best]} final {scores[best][0]:.4f} "
            f"anytime {scores[best][1]:.5f}; in final settings: {grid[best] == used_params}"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "tuning.json", "w") as f:
        json.dump(
            {
                "tuning_seeds": list(TUNING_SEEDS),
                "n_agents": N_AGENTS,
                "repetitions": REPS,
                "budget_factor": final.budget_factor,
                "checkpoints": list(CHECKPOINTS),
                "table": table,
                "selected": selected,
                "runtime_s": round(time.time() - t0, 1),
            },
            f,
            indent=1,
        )
        f.write("\n")
    print(f"tuning: {time.time() - t0:.1f} s -> {args.out / 'tuning.json'}")


if __name__ == "__main__":
    main()
