"""Experiments E1-E7.

Every experiment is a function ``run_eX(settings, jobs)`` that returns a JSON-serialisable
dictionary with the settings, the seeds, the per-run records and the summary tables (mean and
95% Student-t interval over scenarios).  Scenario ``s`` of every experiment uses the user
draw ``numpy.random.default_rng(s)``; all methods of an experiment see the same scenarios
and, where a method needs an initial placement, the same random initial placement.

Random streams: ``numpy.random.default_rng([experiment, scenario, ..., method, repetition])``.
"""

from __future__ import annotations

import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from cfisac_ma import centralized as cen
from cfisac_ma import team
from cfisac_ma.data import Scene, load_scene
from cfisac_ma.objective import Evaluator, Problem, trace_value_at
from cfisac_ma.scenario import (
    SIMULATOR_HOTSPOTS,
    Scenario,
    draw_hotspots,
    draw_users,
    fit_log_distance,
    make_problem,
    make_scenario,
    target_points,
)
from cfisac_ma.stats import mean_ci
from cfisac_ma.structure import estimate_structure, set_functions

#: Order defines the method index used in the random streams; do not reorder.
METHODS = (
    "br",
    "sbr",
    "bandit-team",
    "bandit-diff",
    "greedy",
    "sa",
    "ga",
    "rs",
    "br-diff",
    "greedy-br",
    "br-sat",
    "greedy-sat",
)
METHOD_LABELS = {
    "br": "Best response (team)",
    "br-diff": "Best response (difference utility)",
    "sbr": "Sampled best response (team)",
    "bandit-team": "Bandit learners, team reward",
    "bandit-diff": "Bandit learners, difference reward",
    "greedy": "Greedy (centralised)",
    "sa": "Simulated annealing (centralised)",
    "ga": "Genetic algorithm (centralised)",
    "rs": "Random search (centralised)",
    "greedy-br": "Greedy, then best response",
    "br-sat": "Best response on the saturated-rate potential, then on the utility",
    "greedy-sat": "Greedy on the saturated-rate surrogate (centralised)",
}
DEFAULT_METHODS = ("br", "sbr", "bandit-team", "bandit-diff", "greedy", "sa", "ga", "rs")


@dataclass(frozen=True)
class Settings:
    """Settings shared by the experiments.  ``PASSES`` holds the two configurations used."""

    label: str = "pass2"
    eps: float = 1.0e-4
    lam: float = 0.5
    n_users: int = 15
    n_target_points: int = 16
    #: every method may spend ``budget_factor * N * V`` objective evaluations
    budget_factor: float = 6.0
    sa: dict = field(default_factory=lambda: {"t_start": 100.0, "cooling": 0.99})
    ga: dict = field(default_factory=lambda: {"population": 50, "mutation_rate": 0.1})
    bandit_team: dict = field(default_factory=lambda: {"batch": 32, "learning_rate": 2.0})
    bandit_diff: dict = field(default_factory=lambda: {"batch": 32, "learning_rate": 2.0})
    sbr: dict = field(default_factory=lambda: {"n_random": 16, "n_near": 8, "patience_rounds": 5})
    #: scale the sampled candidate counts with the number of sites (they are given for 345)
    sbr_scale: bool = True
    #: methods added to E2 and E3 after the first pass
    extra_methods: tuple[str, ...] = ("greedy-br",)
    #: surrogate-potential methods added to E6 after the first pass
    e6_extra_methods: tuple[str, ...] = ("br-sat", "greedy-sat")
    # experiment sizes
    n_scenarios: int = 20
    e1_scenarios: int = 10
    n_reps: int = 3
    agents: tuple[int, ...] = (4, 8, 12, 16)
    e1_samples: int = 20_000
    e1_eps: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    e2_agents: tuple[int, ...] = (3, 4, 5)
    e2_candidates: int = 32
    e2_instances: int = 30
    e2_reps: int = 5
    e2_lams: tuple[float, ...] = (1.0, 0.5, 0.0)
    e3_user_counts: tuple[int, ...] = (5, 15, 30, 60)
    #: joint samples per learner in the credit-assignment comparison (0 = skip)
    e3_credit_samples: int = 20_000
    e3_credit_agents: int = 8
    e4_lams: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    e4_agents: int = 8
    e4_scenarios: int = 10
    #: regularisers of the sensitivity check of the lam = 0.5 operating point (empty = skip)
    e4_eps: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    e5_agents: int = 8
    e5_target: float = 0.99
    e7_agents: tuple[int, ...] = (4, 8, 16)
    e7_lams: tuple[float, ...] = (0.5, 1.0)

    def as_dict(self) -> dict:
        return asdict(self)


#: name of the configuration whose results are written to ``results/`` and shown in the README
FINAL_PASS = "pass2"

PASSES: dict[str, Settings] = {
    # First pass: simulated annealing and the genetic algorithm with the hyper-parameters of
    # the simulator's comparison script, untuned learners, no methods or metrics added later.
    "pass1": Settings(
        label="pass1",
        sbr_scale=False,
        extra_methods=(),
        e6_extra_methods=(),
        e1_scenarios=20,
        e1_samples=20_000,
        e3_credit_samples=0,
        e4_eps=(),
    ),
    # Second pass: see docs/notes.md, "What changed between the two passes".  The parameters
    # of the stochastic methods are the grid points selected by scripts/tune_baselines.py.
    "pass2": Settings(
        label="pass2",
        sa={"t_start": 0.003, "t_end": 1e-5, "cooling_fraction": 0.1},
        ga={"population": 50, "mutation_rate": 0.6},
        bandit_team={"batch": 32, "learning_rate": 2.0},
        bandit_diff={"batch": 16, "learning_rate": 2.0},
        sbr={"n_random": 32, "n_near": 16, "patience_rounds": 5},
        e1_samples=10_000,
    ),
}


def quick(settings: Settings) -> Settings:
    """Reduced sizes for smoke tests (a few seconds per experiment)."""
    return replace(
        settings,
        label=settings.label + "-quick",
        n_scenarios=3,
        e1_scenarios=3,
        n_reps=1,
        agents=(4, 8),
        e1_samples=2000,
        e1_eps=(1e-4, 1.0),
        e2_agents=(3,),
        e2_candidates=16,
        e2_instances=3,
        e2_reps=2,
        e3_user_counts=(15, 30),
        e4_lams=(0.0, 0.5, 1.0),
        e4_scenarios=2,
        e4_eps=settings.e4_eps[:2],
        e3_credit_samples=min(settings.e3_credit_samples, 2000),
        e7_agents=(4,),
    )


# ------------------------------------------------------------------------------ infrastructure
_SCENE: Scene | None = None


def get_scene() -> Scene:
    """Scene of the shipped database, loaded once per process."""
    global _SCENE
    if _SCENE is None:
        _SCENE = load_scene()
    return _SCENE


def parallel_map(fn, tasks: list, jobs: int | None = None) -> list:
    """``[fn(t) for t in tasks]`` on ``jobs`` processes; the order of the results is kept."""
    jobs = default_jobs() if jobs is None else jobs
    if jobs <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        return list(pool.map(fn, tasks))


def default_jobs() -> int:
    return max(1, min(10, os.cpu_count() or 1))


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def budget_for(settings: Settings, n_agents: int, n_sites: int) -> int:
    return int(round(settings.budget_factor * n_agents * n_sites))


def scenario_problem(settings: Settings, seed: int, n_users: int | None = None, **kw) -> Problem:
    scene = get_scene()
    sc = make_scenario(
        scene, seed, n_users or settings.n_users, n_target_points=settings.n_target_points
    )
    opts = {"lam": settings.lam, "eps": settings.eps}
    opts.update(kw)
    return make_problem(scene, sc, **opts)


def run_method(
    name: str,
    problem: Problem,
    n_agents: int,
    budget: int,
    rng: np.random.Generator,
    settings: Settings,
    init: np.ndarray,
    warm: bool = False,
) -> team.RunResult:
    """Run one method with an evaluation budget.

    ``init`` is the initial placement of the methods that have one (best response, sampled
    best response, simulated annealing).  With ``warm=True`` the genetic algorithm also
    receives it (as one individual) and simulated annealing uses its warm-start schedule.
    """
    ev = Evaluator(problem, budget)
    n_sites = problem.n_sites
    if name in ("br", "br-diff"):
        reward = "difference" if name == "br-diff" else "team"
        return team.with_restarts(
            lambda e, x, r: team.best_response_dynamics(e, x, r, reward=reward),
            ev,
            n_agents,
            rng,
            init,
            min_evals_per_run=n_sites + 1,
        )
    if name == "sbr":
        params = dict(settings.sbr)
        if settings.sbr_scale:
            for key in ("n_random", "n_near"):
                if params[key] > 0:
                    params[key] = max(1, round(params[key] * n_sites / 345))
        return team.with_restarts(
            lambda e, x, r: team.sampled_best_response(e, x, r, **params),
            ev,
            n_agents,
            rng,
            init,
            min_evals_per_run=n_sites,
        )
    if name == "bandit-team":
        return team.bandit_learning(ev, n_agents, rng, reward="team", **settings.bandit_team)
    if name == "bandit-diff":
        return team.bandit_learning(ev, n_agents, rng, reward="difference", **settings.bandit_diff)
    if name == "greedy":
        return cen.greedy(ev, n_agents)
    if name == "greedy-br":
        first = cen.greedy(ev, n_agents)
        res = team.best_response_dynamics(ev, first.placement)
        res.method = "greedy-br"
        res.n_evals = ev.n_evals
        res.extra = {"greedy_value": first.value, "polish_moves": res.n_moves}
        return res
    if name == "br-sat":
        surrogate = problem.with_options(rate="saturated")
        return team.two_phase_best_response(problem, surrogate, n_agents, budget, rng, init)
    if name == "greedy-sat":
        ev_s = Evaluator(problem.with_options(rate="saturated"), budget - 1)
        first = cen.greedy(ev_s, n_agents)
        value = float(ev.utility(first.placement[None, :])[0])
        return team.RunResult(
            method="greedy-sat",
            placement=first.placement,
            value=value,
            best_placement=first.placement,
            best_value=value,
            n_evals=ev_s.n_evals + 1,
            n_moves=n_agents,
            converged=True,
            trace=[(ev_s.n_evals + 1, value)],
        )
    if name == "sa":
        params = dict(settings.sa)
        if warm and "t_end" in params:
            # warm start: begin at the geometric mean of the two temperatures
            params["t_start"] = float(np.sqrt(params["t_start"] * params["t_end"]))
        return cen.simulated_annealing(ev, init, rng, **params)
    if name == "ga":
        return cen.genetic_algorithm(ev, n_agents, rng, init=init if warm else None, **settings.ga)
    if name == "rs":
        return cen.random_search(ev, n_agents, rng)
    raise ValueError(f"unknown method {name!r}")


def _nanmean(stack: np.ndarray) -> np.ndarray:
    """Column means that ignore ``nan``; a column of only ``nan`` gives ``nan``."""
    ok = ~np.isnan(stack)
    count = ok.sum(axis=0)
    total = np.where(ok, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _grid(budget: int, points: int = 80) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, budget, points)).astype(np.int64))


def _record(name: str, rep: int, res: team.RunResult, problem: Problem) -> dict:
    """Scalar record of one run (raw components of the best placement included)."""
    comp = problem.components(res.best_placement[None, :])
    comm, sens = problem.normalised(comp)
    rec = {
        "method": name,
        "rep": rep,
        "best_value": res.best_value,
        "final_value": res.value,
        "n_evals": res.n_evals,
        "rate_sum": float(comp.rate_sum[0]),
        "rate_min": float(comp.rate_min[0]),
        "logdet": float(comp.logdet[0]),
        "comm_norm": float(comm[0]),
        "sens_norm": float(sens[0]),
        "placement": [int(v) for v in res.best_placement],
    }
    if "first_evals" in res.extra:
        rec.update(
            first_value=res.extra["first_value"],
            first_evals=res.extra["first_evals"],
            first_moves=res.extra["first_moves"],
            first_converged=bool(res.extra["first_converged"]),
            restarts=res.extra["runs"],
        )
    if "polish_moves" in res.extra:
        rec.update(
            greedy_value=res.extra["greedy_value"],
            polish_moves=res.extra["polish_moves"],
            polish_gain=res.best_value - res.extra["greedy_value"],
        )
    return rec


def _run_suite(
    problem: Problem,
    n_agents: int,
    budget: int,
    seed_key: list[int],
    settings: Settings,
    methods: tuple[str, ...],
    reps: int,
    init: np.ndarray | None = None,
    warm: bool = False,
) -> list[tuple[dict, list[tuple[int, float]]]]:
    """All methods and repetitions on one problem; returns ``(record, trace)`` pairs."""
    out = []
    for rep in range(reps):
        if init is None:
            x0 = team.random_placement(
                np.random.default_rng([*seed_key, 999, rep]), problem.n_sites, n_agents
            )
        else:
            x0 = init
        for name in methods:
            if name in ("greedy", "greedy-br", "greedy-sat") and rep > 0:
                continue  # deterministic
            rng = np.random.default_rng([*seed_key, METHODS.index(name), rep])
            res = run_method(name, problem, n_agents, budget, rng, settings, x0, warm=warm)
            out.append((_record(name, rep, res, problem), res.trace))
    return out


def summarise(
    records: list[dict], group: tuple[str, ...], value: str, unit: str = "scenario"
) -> list[dict]:
    """Mean and 95% interval of ``value`` over ``unit`` (repetitions are averaged first)."""
    cells: dict[tuple, dict] = {}
    for r in records:
        if r.get(value) is None:
            continue
        key = tuple(r[g] for g in group)
        cells.setdefault(key, {}).setdefault(r[unit], []).append(float(r[value]))
    rows = []
    for key, per_unit in cells.items():
        means = []
        for v in per_unit.values():
            v = np.asarray(v)
            v = v[~np.isnan(v)]
            if len(v):
                means.append(float(v.mean()))
        rows.append({**dict(zip(group, key, strict=True)), "value": value, **mean_ci(means)})
    return rows


def _finish(result: dict, t0: float) -> dict:
    result["runtime_s"] = round(time.time() - t0, 2)
    result["machine"] = machine_info()
    return result


# ------------------------------------------------------------------------------ E1 structure
def _e1_task(args) -> list[dict]:
    settings, seed = args
    rows = []
    for eps in settings.e1_eps:
        problem = scenario_problem(settings, seed, eps=eps)
        rng = np.random.default_rng([1, seed])
        rx = rng.choice(problem.n_sites, size=4, replace=False)
        functions = set_functions(problem, rx_sites=rx)
        for j, (name, (f, n_ground)) in enumerate(functions.items()):
            if eps != settings.eps and name not in ("sensing_joint", "utility"):
                continue  # the eps sweep covers the two functions that are not submodular
            est = estimate_structure(
                f, n_ground, np.random.default_rng([1, seed, j]), n_samples=settings.e1_samples
            )
            rows.append({"scenario": seed, "eps": eps, "function": name, **est.as_dict()})
    return rows


def run_e1(settings: Settings, jobs: int | None = None) -> dict:
    """E1: submodularity violations, submodularity ratio, curvature of the objective parts."""
    t0 = time.time()
    tasks = [(settings, s) for s in range(settings.e1_scenarios)]
    rows = [r for part in parallel_map(_e1_task, tasks, jobs) for r in part]
    keys = (
        "violation_rate",
        "max_violation",
        "monotonicity_violation_rate",
        "submodularity_ratio_min",
        "submodularity_ratio_p05",
        "dr_ratio_min",
        "dr_ratio_p05",
        "curvature_max",
        "curvature_p95",
        "greedy_factor",
        "equilibrium_factor",
    )
    summary = []
    for k in keys:
        summary += summarise(rows, ("function", "eps"), k)
    # Extremes over all scenarios (the quantities are defined as extremes over sets).
    extremes = []
    for fn in sorted({r["function"] for r in rows}):
        for eps in sorted({r["eps"] for r in rows if r["function"] == fn}):
            sel = [r for r in rows if r["function"] == fn and r["eps"] == eps]
            # a scenario without any sample of positive gain contributes no estimate (nan)
            gamma = float(np.nanmin([r["submodularity_ratio_min"] for r in sel]))
            gamma_d = float(np.nanmin([r["dr_ratio_min"] for r in sel]))
            alpha = float(np.nanmax([r["curvature_max"] for r in sel]))
            gc, gd = min(gamma, 1.0), min(gamma_d, 1.0)
            g = gc * gd**2
            extremes.append(
                {
                    "function": fn,
                    "eps": eps,
                    "violation_rate_max": max(r["violation_rate"] for r in sel),
                    "submodularity_ratio_min": gamma,
                    "dr_ratio_min": gamma_d,
                    "curvature_max": alpha,
                    "greedy_factor": float((1 - np.exp(-alpha * gc)) / alpha)
                    if alpha > 1e-12
                    else gc,
                    "equilibrium_factor": g / (1 + g),
                }
            )
    # Share of violating triples by the size of the smaller set S (reference eps).
    by_size = []
    for fn in sorted({r["function"] for r in rows}):
        sel = [r for r in rows if r["function"] == fn and r["eps"] == settings.eps]
        for size in range(len(sel[0].get("violation_rate_by_base_size", ()))):
            ci = mean_ci([r["violation_rate_by_base_size"][size] for r in sel])
            by_size.append({"function": fn, "base_size": size, **ci})
    return _finish(
        {
            "experiment": "e1_structure",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.e1_scenarios)),
            "rows": rows,
            "summary": summary,
            "extremes": extremes,
            "violations_by_base_size": by_size,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E2 small instances
E2_METHODS = ("br", "br-diff", "sbr", "bandit-team", "bandit-diff", "greedy", "sa", "ga", "rs")


def _e2_task(args) -> dict:
    settings, seed, n_agents = args
    scene = get_scene()
    sub_rng = np.random.default_rng([2, seed, n_agents])
    sites = np.sort(sub_rng.choice(scene.n_sites, size=settings.e2_candidates, replace=False))
    sc = make_scenario(scene, seed, settings.n_users, n_target_points=settings.n_target_points)
    base = make_problem(scene, sc, lam=settings.lam, eps=settings.eps, sites=sites)
    n_sites = base.n_sites
    budget = budget_for(settings, n_agents, n_sites)
    instances, runs = [], []
    # Ground truth for all weights at once: utilities of all subsets and all equilibria.
    subsets, comm, sens = cen.exhaustive_terms(base, n_agents)
    all_values = np.stack([lam * comm + (1.0 - lam) * sens for lam in settings.e2_lams])
    all_ne = cen.nash_equilibria(subsets, all_values, n_sites)
    for j, lam in enumerate(settings.e2_lams):
        problem = base.with_options(lam=lam)
        values, is_ne = all_values[j], all_ne[j]
        opt = float(values.max())
        ne_values = values[is_ne]
        instances.append(
            {
                "scenario": seed,
                "n_agents": n_agents,
                "lam": lam,
                "opt": opt,
                "n_subsets": int(len(values)),
                "n_equilibria": int(is_ne.sum()),
                "worst_ne_ratio": float(ne_values.min() / opt),
                "mean_ne_ratio": float(ne_values.mean() / opt),
                "random_ratio": float(values.mean() / opt),
                "sites": [int(v) for v in sites],
            }
        )
        # Single best-response runs (no restarts): which equilibrium is reached, at what cost.
        for rep in range(settings.e2_reps):
            x0 = team.random_placement(
                np.random.default_rng([2, seed, n_agents, 999, rep]), n_sites, n_agents
            )
            single = {}
            for reward in ("team", "difference"):
                res = team.best_response_dynamics(Evaluator(problem), x0, reward=reward)
                single[reward] = res
                key = np.sort(res.placement)[None, :]
                runs.append(
                    {
                        "scenario": seed,
                        "n_agents": n_agents,
                        "lam": lam,
                        "rep": rep,
                        "method": "br-single" if reward == "team" else "br-diff-single",
                        "ratio": res.value / opt,
                        "n_evals": res.n_evals,
                        "n_moves": res.n_moves,
                        "is_equilibrium": bool(is_ne[cen.subset_rank(key, n_sites)[0]])
                        if len(np.unique(res.placement)) == n_agents
                        else False,
                        "is_optimal": bool(res.value >= opt - 1e-9),
                    }
                )
            runs[-1]["same_as_team"] = bool(
                np.array_equal(single["team"].placement, single["difference"].placement)
            )
        # All methods with the common budget.
        suite = _run_suite(
            problem,
            n_agents,
            budget,
            [2, seed, n_agents, int(round(lam * 100))],
            settings,
            E2_METHODS + settings.extra_methods,
            settings.e2_reps,
        )
        for rec, _ in suite:
            runs.append(
                {
                    "scenario": seed,
                    "n_agents": n_agents,
                    "lam": lam,
                    "rep": rec["rep"],
                    "method": rec["method"],
                    "ratio": rec["best_value"] / opt,
                    "n_evals": rec["n_evals"],
                    "is_optimal": bool(rec["best_value"] >= opt - 1e-9),
                }
            )
    return {"instances": instances, "runs": runs, "budget": budget}


def run_e2(settings: Settings, jobs: int | None = None) -> dict:
    """E2: small instances with the brute-force optimum and all pure Nash equilibria."""
    t0 = time.time()
    tasks = [(settings, s, n) for n in settings.e2_agents for s in range(settings.e2_instances)]
    parts = parallel_map(_e2_task, tasks, jobs)
    instances = [r for p in parts for r in p["instances"]]
    runs = [r for p in parts for r in p["runs"]]
    summary = []
    for k in ("worst_ne_ratio", "mean_ne_ratio", "n_equilibria", "random_ratio"):
        summary += summarise(instances, ("lam", "n_agents"), k)
    for k in ("ratio", "is_optimal", "n_evals"):
        summary += summarise(runs, ("lam", "n_agents", "method"), k)
    minima = []
    for lam in settings.e2_lams:
        for n in settings.e2_agents:
            inst = [r for r in instances if r["lam"] == lam and r["n_agents"] == n]
            minima.append(
                {
                    "lam": lam,
                    "n_agents": n,
                    "method": "worst-equilibrium",
                    "min_ratio": min(r["worst_ne_ratio"] for r in inst),
                }
            )
            for m in sorted({r["method"] for r in runs}):
                sel = [
                    r["ratio"]
                    for r in runs
                    if r["lam"] == lam and r["n_agents"] == n and r["method"] == m
                ]
                minima.append({"lam": lam, "n_agents": n, "method": m, "min_ratio": min(sel)})
    same = [r["same_as_team"] for r in runs if "same_as_team" in r]
    return _finish(
        {
            "experiment": "e2_small_instances",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.e2_instances)),
            "budgets": {
                str(n): parts[i * settings.e2_instances]["budget"]
                for i, n in enumerate(settings.e2_agents)
            },
            "instances": instances,
            "runs": runs,
            "summary": summary,
            "minima": minima,
            "difference_utility_same_endpoint_fraction": float(np.mean(same)),
        },
        t0,
    )


# ------------------------------------------------------------------------------ E3 full problem
def _full_task(args) -> dict:
    """All methods on one full-size problem (used by E3 and E6)."""
    settings, exp_id, seed, n_agents, n_users, rate, methods = args
    problem = scenario_problem(settings, seed, n_users, rate=rate)
    budget = budget_for(settings, n_agents, problem.n_sites)
    suite = _run_suite(
        problem,
        n_agents,
        budget,
        [exp_id, seed, n_agents, n_users],
        settings,
        methods,
        settings.n_reps,
    )
    best_known = max(rec["best_value"] for rec, _ in suite)
    grid = _grid(budget)
    records, curves = [], {}
    for rec, trace in suite:
        rec.update(
            scenario=seed,
            n_agents=n_agents,
            n_users=n_users,
            rate=rate,
            best_known=best_known,
            ratio=rec["best_value"] / best_known if best_known > 0 else np.nan,
        )
        hit = [e for e, v in trace if v >= 0.99 * best_known]
        rec["evals_to_99"] = hit[0] if hit else None
        hit = [e for e, v in trace if v >= 0.999 * best_known]
        rec["evals_to_999"] = hit[0] if hit else None
        rec["found_best_known"] = float(rec["best_value"] >= best_known - 1e-9)
        records.append(rec)
        if best_known > 0:
            # Before a run has evaluated its first placement it has nothing to deploy: the
            # curve is 0 there, the utility of the empty placement.  (Only the surrogate
            # methods of E6 start late; all other traces start at the first evaluation.)
            curve = np.nan_to_num(trace_value_at(trace, grid), nan=0.0) / best_known
        else:
            curve = np.full(len(grid), np.nan)
        curves.setdefault(rec["method"], []).append(curve)
    mean_curves = {m: _nanmean(np.stack(c)).tolist() for m, c in curves.items()}
    return {"records": records, "grid": grid.tolist(), "curves": mean_curves, "budget": budget}


def _curve_summary(parts: list[dict], keys: list[tuple]) -> list[dict]:
    """Mean and 95% interval over scenarios of the per-scenario mean curves."""
    out = []
    for key in sorted(set(keys)):
        sel = [p for p, k in zip(parts, keys, strict=True) if k == key]
        grid = sel[0]["grid"]
        for m in sel[0]["curves"]:
            stack = np.array([p["curves"][m] for p in sel], dtype=np.float64)
            cis = [mean_ci(stack[:, j]) for j in range(stack.shape[1])]
            out.append(
                {
                    "key": list(key),
                    "method": m,
                    "evals": grid,
                    "budget": sel[0]["budget"],
                    "mean": [c["mean"] for c in cis],
                    "lo": [c["lo"] for c in cis],
                    "hi": [c["hi"] for c in cis],
                }
            )
    return out


def _full_summary(records: list[dict], group: tuple[str, ...]) -> list[dict]:
    summary = []
    for k in (
        "best_value",
        "ratio",
        "found_best_known",
        "rate_sum",
        "rate_min",
        "logdet",
        "n_evals",
        "evals_to_99",
        "evals_to_999",
        "first_value",
        "first_evals",
        "first_moves",
        "first_converged",
        "polish_moves",
        "polish_gain",
    ):
        summary += summarise(records, group, k)
    # fraction of runs that reached 99% / 99.9% of the best-known utility within the budget
    for r in records:
        r["reached_99"] = float(r["evals_to_99"] is not None)
        r["reached_999"] = float(r["evals_to_999"] is not None)
    summary += summarise(records, group, "reached_99")
    summary += summarise(records, group, "reached_999")
    return summary


def _credit_task(args) -> dict:
    """Both learners with the same number of joint samples (credit-assignment comparison)."""
    settings, seed = args
    n_agents = settings.e3_credit_agents
    problem = scenario_problem(settings, seed)
    x0 = team.random_placement(np.random.default_rng([3, seed, 70]), problem.n_sites, n_agents)
    reference = team.best_response_dynamics(Evaluator(problem), x0).value
    grid = np.unique(np.round(np.geomspace(64, settings.e3_credit_samples, 40)).astype(np.int64))
    out = {"scenario": seed, "reference": reference, "grid": grid.tolist(), "rewards": {}}
    for j, reward in enumerate(team.REWARDS):
        params = settings.bandit_team if reward == "team" else settings.bandit_diff
        ev = Evaluator(problem)
        res = team.bandit_learning(
            ev,
            n_agents,
            np.random.default_rng([3, seed, 71, j]),
            reward=reward,
            max_iterations=settings.e3_credit_samples // params["batch"],
            **params,
        )
        samples = params["batch"] * (np.arange(len(res.extra["batch_mean"])) + 1)
        curve = np.interp(grid, samples, res.extra["batch_mean"], left=np.nan, right=np.nan)
        out["rewards"][reward] = {
            "curve": (curve / reference).tolist(),
            "n_evals": res.n_evals,
            "greedy_policy_ratio": res.value / reference,
            "best_sampled_ratio": res.best_value / reference,
        }
    return out


def _credit_summary(parts: list[dict]) -> dict:
    grid = parts[0]["grid"]
    out = {"samples": grid, "n_scenarios": len(parts), "rewards": {}}
    for reward in team.REWARDS:
        stack = np.array([p["rewards"][reward]["curve"] for p in parts], dtype=np.float64)
        cis = [mean_ci(stack[:, j]) for j in range(stack.shape[1])]
        out["rewards"][reward] = {
            "mean": [c["mean"] for c in cis],
            "lo": [c["lo"] for c in cis],
            "hi": [c["hi"] for c in cis],
            "n_evals": mean_ci([p["rewards"][reward]["n_evals"] for p in parts]),
            "greedy_policy_ratio": mean_ci(
                [p["rewards"][reward]["greedy_policy_ratio"] for p in parts]
            ),
            "best_sampled_ratio": mean_ci(
                [p["rewards"][reward]["best_sampled_ratio"] for p in parts]
            ),
        }
    return out


def run_e3(settings: Settings, jobs: int | None = None) -> dict:
    """E3: full problem, utility versus objective evaluations for several team sizes."""
    t0 = time.time()
    methods = DEFAULT_METHODS + settings.extra_methods
    tasks, keys = [], []
    for n in settings.agents:
        for s in range(settings.n_scenarios):
            tasks.append((settings, 3, s, n, settings.n_users, "sum", methods))
            keys.append((n, settings.n_users))
    n_mid = settings.agents[min(1, len(settings.agents) - 1)]
    for k in settings.e3_user_counts:
        if k == settings.n_users:
            continue
        for s in range(settings.n_scenarios):
            tasks.append((settings, 3, s, n_mid, k, "sum", methods))
            keys.append((n_mid, k))
    parts = parallel_map(_full_task, tasks, jobs)
    records = [r for p in parts for r in p["records"]]
    result = {
        "experiment": "e3_full_problem",
        "settings": settings.as_dict(),
        "scenario_seeds": list(range(settings.n_scenarios)),
        "budgets": {str(n): budget_for(settings, n, get_scene().n_sites) for n in settings.agents},
        "runs": records,
        "summary": _full_summary(records, ("n_agents", "n_users", "method")),
        "curves": _curve_summary(parts, keys),
    }
    if settings.e3_credit_samples > 0:
        credit = parallel_map(
            _credit_task, [(settings, s) for s in range(settings.n_scenarios)], jobs
        )
        result["credit_assignment"] = _credit_summary(credit)
    return _finish(result, t0)


# ------------------------------------------------------------------------------ E4 trade-off
E4_METHODS = ("br", "sbr", "greedy", "sa", "ga", "rs")


def _e4_task(args) -> list[dict]:
    settings, seed, lam = args
    problem = scenario_problem(settings, seed, lam=lam)
    n_agents = settings.e4_agents
    budget = budget_for(settings, n_agents, problem.n_sites)
    suite = _run_suite(
        problem, n_agents, budget, [4, seed, int(round(lam * 100))], settings, E4_METHODS, 1
    )
    out = []
    for rec, _ in suite:
        rec.update(scenario=seed, lam=lam, n_agents=n_agents)
        out.append(rec)
    return out


def _e4_eps_task(args) -> dict:
    """Best-response placement for another regulariser, measured with the reference one."""
    settings, seed, eps = args
    n_agents = settings.e4_agents
    reference = scenario_problem(settings, seed)
    problem = reference.with_options(eps=eps)
    x0 = team.random_placement(np.random.default_rng([4, seed, 80]), problem.n_sites, n_agents)
    res = team.best_response_dynamics(Evaluator(problem), x0)
    own_comm, own_sens = problem.normalised(problem.components(res.placement[None, :]))
    comp = reference.components(res.placement[None, :])
    return {
        "scenario": seed,
        "eps": eps,
        "method": "br-single",
        "rate_sum": float(comp.rate_sum[0]),
        "logdet_at_reference_eps": float(comp.logdet[0]),
        "utility_at_reference_eps": float(reference.utility(res.placement[None, :])[0]),
        "comm_norm_own_eps": float(own_comm[0]),
        "sens_norm_own_eps": float(own_sens[0]),
        "sensing_share_of_utility": float(
            (1 - problem.lam)
            * own_sens[0]
            / (problem.lam * own_comm[0] + (1 - problem.lam) * own_sens[0])
        ),
    }


def run_e4(settings: Settings, jobs: int | None = None) -> dict:
    """E4: sweep of the weight ``lam``; achieved (sum rate, sensing) pairs per method."""
    t0 = time.time()
    tasks = [(settings, s, lam) for lam in settings.e4_lams for s in range(settings.e4_scenarios)]
    records = [r for part in parallel_map(_e4_task, tasks, jobs) for r in part]
    summary = []
    for k in ("rate_sum", "logdet", "comm_norm", "sens_norm", "best_value"):
        summary += summarise(records, ("lam", "method"), k)
    result = {
        "experiment": "e4_tradeoff",
        "settings": settings.as_dict(),
        "scenario_seeds": list(range(settings.e4_scenarios)),
        "runs": records,
        "summary": summary,
    }
    if settings.e4_eps:
        eps_tasks = [
            (settings, s, e) for e in settings.e4_eps for s in range(settings.e4_scenarios)
        ]
        eps_rows = parallel_map(_e4_eps_task, eps_tasks, jobs)
        result["eps_sensitivity"] = eps_rows
        for k in (
            "rate_sum",
            "logdet_at_reference_eps",
            "utility_at_reference_eps",
            "comm_norm_own_eps",
            "sens_norm_own_eps",
            "sensing_share_of_utility",
        ):
            summary += summarise(eps_rows, ("eps", "method"), k)
    return _finish(result, t0)


# ------------------------------------------------------------------------------ E5 adaptation
E5_WARM = ("br", "sbr", "sa", "ga")
E5_COLD = ("br", "sbr", "greedy", "sa", "ga", "rs")
E5_CHANGES = {"one hotspot moves": (0, 1, 2), "all hotspots move": ()}


def _e5_task(args) -> list[dict]:
    settings, seed, change = args
    scene = get_scene()
    n_agents = settings.e5_agents
    old = scenario_problem(settings, seed)
    n_sites = old.n_sites
    budget = budget_for(settings, n_agents, n_sites)
    # Old equilibrium: best response from a random placement on the old users.
    x0 = team.random_placement(np.random.default_rng([5, seed, 0]), n_sites, n_agents)
    old_eq = team.best_response_dynamics(Evaluator(old), x0).placement
    # New users: move one hotspot (chosen at random) or all of them.
    rng = np.random.default_rng([5, seed, 1, list(E5_CHANGES).index(change)])
    keep = E5_CHANGES[change]
    if keep:
        moved = int(rng.integers(len(SIMULATOR_HOTSPOTS)))
        keep = tuple(i for i in keep if i != moved)
    hotspots = draw_hotspots(scene, rng, keep=keep)
    users = draw_users(scene, rng, settings.n_users, hotspots)
    sc = Scenario(users=users, targets=target_points(settings.n_target_points), hotspots=hotspots)
    new = make_problem(scene, sc, lam=settings.lam, eps=settings.eps)
    key = [5, seed, list(E5_CHANGES).index(change)]
    warm = _run_suite(
        new, n_agents, budget, key + [1], settings, E5_WARM, settings.n_reps, init=old_eq, warm=True
    )
    cold = _run_suite(new, n_agents, budget, key + [0], settings, E5_COLD, settings.n_reps)
    # Reference: best utility found by any run, plus a long multi-start best response.
    long_run = run_method(
        "br", new, n_agents, 4 * budget, np.random.default_rng(key + [7]), settings, x0
    )
    reference = max([long_run.best_value] + [rec["best_value"] for rec, _ in warm + cold])
    stale = float(new.utility(old_eq[None, :])[0])
    out = []
    for start, suite in (("warm", warm), ("cold", cold)):
        for rec, trace in suite:
            hit = [e for e, v in trace if v >= settings.e5_target * reference]
            out.append(
                {
                    "scenario": seed,
                    "change": change,
                    "start": start,
                    "method": rec["method"],
                    "rep": rec["rep"],
                    "reference": reference,
                    "stale_ratio": stale / reference,
                    "ratio": rec["best_value"] / reference,
                    "reached": float(bool(hit)),
                    "evals_to_target": hit[0] if hit else None,
                    "evals_censored": hit[0] if hit else budget,
                    "budget": budget,
                }
            )
    return out


def run_e5(settings: Settings, jobs: int | None = None) -> dict:
    """E5: adaptation to moved user hotspots, warm start from the old equilibrium vs cold."""
    t0 = time.time()
    tasks = [(settings, s, c) for c in E5_CHANGES for s in range(settings.n_scenarios)]
    records = [r for part in parallel_map(_e5_task, tasks, jobs) for r in part]
    summary = []
    for k in ("evals_censored", "evals_to_target", "reached", "ratio", "stale_ratio"):
        summary += summarise(records, ("change", "start", "method"), k)
    return _finish(
        {
            "experiment": "e5_adaptation",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "runs": records,
            "summary": summary,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E6 max-min
def run_e6(settings: Settings, jobs: int | None = None) -> dict:
    """E6: max-min rate variant of the utility on the full problem."""
    t0 = time.time()
    tasks, keys = [], []
    methods = DEFAULT_METHODS + settings.e6_extra_methods
    for n in settings.agents:
        for s in range(settings.n_scenarios):
            tasks.append((settings, 6, s, n, settings.n_users, "maxmin", methods))
            keys.append((n, settings.n_users))
    parts = parallel_map(_full_task, tasks, jobs)
    records = [r for p in parts for r in p["records"]]
    for r in records:
        r["min_rate_positive"] = float(r["rate_min"] > 0)
    summary = _full_summary(records, ("n_agents", "n_users", "method"))
    summary += summarise(records, ("n_agents", "n_users", "method"), "min_rate_positive")
    summary += summarise(records, ("n_agents", "n_users", "method"), "comm_norm")
    return _finish(
        {
            "experiment": "e6_maxmin",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "budgets": {
                str(n): budget_for(settings, n, get_scene().n_sites) for n in settings.agents
            },
            "runs": records,
            "summary": summary,
            "curves": _curve_summary(parts, keys),
        },
        t0,
    )


# ------------------------------------------------------------------------------ E7 channel model
E7_CHANNELS = ("raytraced", "free_space", "log_distance")
E7_METHODS = ("br", "greedy")


def _e7_task(args) -> list[dict]:
    settings, seed, n_agents, lam, exponent = args
    scene = get_scene()
    sc = make_scenario(scene, seed, settings.n_users, n_target_points=settings.n_target_points)
    truth = make_problem(scene, sc, lam=lam, eps=settings.eps)
    n_sites = truth.n_sites
    budget = budget_for(settings, n_agents, n_sites)
    rand = np.argsort(np.random.default_rng([7, seed, n_agents, 5]).random((500, n_sites)), axis=1)
    random_value = float(truth.utility(rand[:, :n_agents], assume_unique=True).mean())
    x0 = team.random_placement(np.random.default_rng([7, seed, n_agents, 999]), n_sites, n_agents)
    placements = {}
    for channel in E7_CHANNELS:
        model = make_problem(
            scene, sc, lam=lam, eps=settings.eps, channel=channel, exponent=exponent
        )
        for method in E7_METHODS:
            rng = np.random.default_rng([7, seed, n_agents, E7_CHANNELS.index(channel)])
            res = run_method(method, model, n_agents, budget, rng, settings, x0)
            placements[(channel, method)] = res.best_placement
    out = []
    for (channel, method), placement in placements.items():
        comp = truth.components(placement[None, :])
        comm, sens = truth.normalised(comp)
        ref = placements[("raytraced", method)]
        ref_comp = truth.components(ref[None, :])
        ref_comm, ref_sens = truth.normalised(ref_comp)
        u = float(lam * comm[0] + (1 - lam) * sens[0])
        u_ref = float(lam * ref_comm[0] + (1 - lam) * ref_sens[0])
        out.append(
            {
                "scenario": seed,
                "n_agents": n_agents,
                "lam": lam,
                "channel": channel,
                "method": method,
                "utility": u,
                "utility_ref": u_ref,
                "gap": u_ref - u,
                "relative_gap": (u_ref - u) / u_ref,
                "gap_rate_part": float(lam * (ref_comm[0] - comm[0])),
                "gap_sensing_part": float((1 - lam) * (ref_sens[0] - sens[0])),
                "rate_sum": float(comp.rate_sum[0]),
                "rate_sum_ref": float(ref_comp.rate_sum[0]),
                "rate_sum_loss_fraction": float(1 - comp.rate_sum[0] / ref_comp.rate_sum[0]),
                "logdet": float(comp.logdet[0]),
                "logdet_ref": float(ref_comp.logdet[0]),
                "overlap": len(set(placement.tolist()) & set(ref.tolist())) / n_agents,
                "random_utility": random_value,
                "retained_gain": (u - random_value) / (u_ref - random_value),
                "placement": [int(v) for v in placement],
            }
        )
    return out


def run_e7(settings: Settings, jobs: int | None = None) -> dict:
    """E7: placements optimised on a statistical channel, evaluated on the ray-traced one."""
    t0 = time.time()
    fit = fit_log_distance(get_scene())
    tasks = [
        (settings, s, n, lam, fit["exponent"])
        for lam in settings.e7_lams
        for n in settings.e7_agents
        for s in range(settings.n_scenarios)
    ]
    records = [r for part in parallel_map(_e7_task, tasks, jobs) for r in part]
    summary = []
    for k in (
        "utility",
        "gap",
        "relative_gap",
        "gap_rate_part",
        "gap_sensing_part",
        "rate_sum_loss_fraction",
        "overlap",
        "retained_gain",
    ):
        summary += summarise(records, ("lam", "n_agents", "channel", "method"), k)
    return _finish(
        {
            "experiment": "e7_channel_model",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "fit": fit,
            "runs": records,
            "summary": summary,
        },
        t0,
    )


EXPERIMENTS = {
    "e1": run_e1,
    "e2": run_e2,
    "e3": run_e3,
    "e4": run_e4,
    "e5": run_e5,
    "e6": run_e6,
    "e7": run_e7,
}
