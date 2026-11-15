"""Experiments E8-E12: blockage-aware sensing and the operation of a deployed network.

The conventions are those of :mod:`cfisac_ma.experiments`: every experiment is a function
``run_eX(settings, jobs)`` that returns a JSON-serialisable dictionary with the settings,
the per-run records and summary tables (mean and 95% Student-t interval over scenarios).
Scenario ``s`` uses the user draw ``numpy.random.default_rng(s)`` of E1-E7; planning points,
planning users, UAV paths and algorithmic randomness use
``numpy.random.default_rng([stream, scenario, ...])``, so every task is reproducible on its
own and the results do not depend on the number of worker processes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from cfisac_ma import aerial, team
from cfisac_ma import centralized as cen
from cfisac_ma import experiments as ex
from cfisac_ma import operation as op
from cfisac_ma.data import default_data_dir
from cfisac_ma.objective import Evaluator
from cfisac_ma.scenario import (
    SIMULATOR_HOTSPOTS,
    Scenario,
    draw_users,
    make_problem,
    make_scenario,
    target_points,
    user_snr,
)
from cfisac_ma.stats import mean_ci
from cfisac_ma.structure import estimate_structure, set_functions

#: random streams (first entry of the seed sequence)
STREAM_PLANNING_POINTS = 80
STREAM_PLANNING_USERS = 81
STREAM_PATHS = 82
STREAM_RANDOM_DEPLOYMENT = 83

TUNING_SEEDS = (1000, 1001, 1002, 1003, 1004)


@dataclass(frozen=True)
class OpSettings:
    """Settings of E8-E12.  ``PASSES`` holds the two configurations used."""

    label: str = "pass2"
    eps: float = 1.0e-4
    lam: float = 0.5
    rho: float = 1.0e12
    n_users: int = 15
    #: flight level (index into 10, 20, 30, 40 m) of each UAV
    levels: tuple[int, ...] = (0, 1, 2, 3)
    planning_points_per_level: int = 16
    planning_users: int = 60
    #: energy cost of a transmitting / receiving AP in units of the team utility
    c_tx: float = 0.004
    c_rx: float = 0.002
    slot_s: float = 0.5
    n_slots: int = 240
    speed: float = 10.0
    n_scenarios: int = 20
    #: simulated annealing over mode profiles from a cold start
    sa: dict = field(
        default_factory=lambda: {"t_start": 0.003, "t_end": 1e-5, "cooling_fraction": 0.1}
    )
    #: cold schedules for the deployment sizes of E11 (key: number of APs as a string);
    #: sizes without an entry use ``sa``
    sa_e11: dict = field(default_factory=dict)
    #: constant temperature of the warm-started annealing of the online controller
    sa_warm_temperature: float = 0.0
    lll: dict = field(default_factory=lambda: {"tau_start": 3e-3, "tau_end": 1e-4, "fraction": 0.5})
    #: random starts of the reference solver (besides the warm start and all-off)
    reference_random_starts: int = 2
    # E8
    e8_agents: tuple[int, ...] = (4, 8, 16)
    e8_structure_scenarios: int = 10
    e8_samples: int = 10_000
    # E9
    e9_agents: tuple[int, ...] = (8, 10, 12)
    e9_lams: tuple[float, ...] = (0.5, 0.0, 1.0)
    e9_budget_factor: int = 60
    e9_reps: int = 3
    # E10
    e10_agents: int = 24
    e10_speeds: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0)
    e10_subrounds: tuple[int, ...] = (1, 2, 8)
    e10_delays: tuple[int, ...] = (0, 1, 2, 4, 8)
    e10_clusters: tuple[int, ...] = (4, 8)
    e10_period: int = 20
    # E11
    e11_agents: tuple[int, ...] = (12, 24, 48, 96)
    e11_clusters: tuple[int, ...] = (4, 8)
    e11_sa_budget_factor: int = 90
    # E12
    e12_agents: int = 24
    e12_failures: tuple[int, ...] = (1, 2, 4)
    e12_reps: int = 3
    #: sensitivity sweep of the cost scale (multiples of ``c_tx``, ``c_rx``)
    cost_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

    def as_dict(self) -> dict:
        return asdict(self)


FINAL_PASS = "pass2"

PASSES: dict[str, OpSettings] = {
    # First pass: the start values of the costs, the cold annealing schedule that was tuned
    # for the placement problem (E1-E7), no temperature in the online annealing.
    "pass1": OpSettings(label="pass1"),
    # Second pass: see docs/notes.md, Part II.
    "pass2": OpSettings(
        label="pass2",
        c_tx=0.016,
        c_rx=0.008,
        sa={"t_start": 0.01, "t_end": 0.001, "cooling_fraction": 1.0},
        sa_e11={
            "12": {"t_start": 0.01, "t_end": 0.001, "cooling_fraction": 0.5},
            "24": {"t_start": 0.003, "t_end": 0.001, "cooling_fraction": 0.5},
            "48": {"t_start": 0.003, "t_end": 1e-05, "cooling_fraction": 1.0},
            "96": {"t_start": 0.03, "t_end": 0.0001, "cooling_fraction": 0.5},
        },
        sa_warm_temperature=0.001,
        lll={"tau_start": 0.01, "tau_end": 0.001, "fraction": 0.5},
    ),
}


def quick(settings: OpSettings) -> OpSettings:
    """Reduced sizes for smoke tests."""
    return replace(
        settings,
        label=settings.label + "-quick",
        n_scenarios=3,
        n_slots=40,
        e8_agents=(4,),
        e8_structure_scenarios=2,
        e8_samples=1000,
        e9_agents=(6,),
        e9_reps=1,
        e10_agents=12,
        e10_speeds=(10.0, 40.0),
        e10_clusters=(4,),
        e11_agents=(12, 24),
        e12_agents=12,
        e12_failures=(1, 2),
        e12_reps=1,
        cost_scales=(0.0, 1.0, 4.0),
    )


# ------------------------------------------------------------------------------ infrastructure
_LOS: aerial.AerialLos | None = None


def get_los() -> aerial.AerialLos:
    global _LOS
    if _LOS is None:
        _LOS = aerial.load_aerial_los()
        if not np.array_equal(_LOS.site_ids, ex.get_scene().site_ids):
            raise ValueError("line-of-sight tables and path-gain database disagree in the sites")
    return _LOS


def placement_settings() -> ex.Settings:
    """Settings of the placement optimisers (budget, restarts) used for deployments."""
    return ex.PASSES[ex.FINAL_PASS]


def planning_problem(settings: OpSettings, seed: int):
    """Blockage-aware placement problem used to plan a deployment for scenario ``seed``.

    Planning points: ``planning_points_per_level`` free cells per flight level; planning
    users: ``planning_users`` draws from the simulator's hotspot mixture (not the users of
    the scenario itself, which the planner does not know).  Joint roles, weight ``lam``.
    """
    scene, los = ex.get_scene(), get_los()
    points = aerial.planning_points(
        los,
        np.random.default_rng([STREAM_PLANNING_POINTS, seed]),
        settings.planning_points_per_level,
    )
    users = draw_users(
        scene, np.random.default_rng([STREAM_PLANNING_USERS, seed]), settings.planning_users
    )
    scenario = Scenario(users=users, targets=points, hotspots=SIMULATOR_HOTSPOTS)
    problem = make_problem(
        scene, scenario, lam=settings.lam, eps=settings.eps, rho=settings.rho, blockage=los
    )
    return problem, points


def deploy(settings: OpSettings, seed: int, n_agents: int, kind: str = "planned") -> dict:
    """Sites of a deployment and the sensing normaliser ``S_ref`` on the planning points.

    ``planned``: greedy selection followed by best response on :func:`planning_problem`;
    ``random``: uniformly drawn distinct valid sites.
    """
    problem, points = planning_problem(settings, seed)
    if kind == "planned":
        ev = Evaluator(problem)
        first = cen.greedy(ev, n_agents)
        sites = team.best_response_dynamics(ev, first.placement).placement
    elif kind == "random":
        rng = np.random.default_rng([STREAM_RANDOM_DEPLOYMENT, seed])
        sites = rng.choice(problem.n_sites, size=n_agents, replace=False)
    else:
        raise ValueError("kind must be 'planned' or 'random'")
    sites = np.sort(sites)
    comp = problem.components(sites[None, :])
    return {
        "sites": sites,
        "s_ref": float(comp.logdet[0] - problem.logdet_empty),
        "planning_utility": float(problem.utility(sites[None, :])[0]),
        "planning_points": points,
    }


def paths(settings: OpSettings, seed: int, levels: tuple[int, ...], speed: float, n_slots: int):
    """UAV positions ``[n_slots, Q, 3]`` of scenario ``seed`` (same paths for every speed)."""
    longest = max(settings.e10_speeds + (settings.speed, speed))
    return aerial.target_tracks(
        get_los(), [STREAM_PATHS, seed], levels, speed, n_slots, settings.slot_s, longest
    )


def make_episode(
    settings: OpSettings,
    seed: int,
    n_agents: int,
    *,
    speed: float | None = None,
    n_slots: int | None = None,
    n_users: int | None = None,
    levels: tuple[int, ...] | None = None,
    deployment: str = "planned",
    normalise_as: str | None = None,
    lam: float | None = None,
    costs: tuple[float, float] | None = None,
) -> op.Episode:
    """Episode of scenario ``seed``: deployment, users, UAV tracks, normalisers.

    ``normalise_as`` names the deployment whose ``R_ref`` and ``S_ref`` are used (default:
    the episode's own deployment), so that utilities of two deployments can be compared.
    """
    scene, los = ex.get_scene(), get_los()
    dep = deploy(settings, seed, n_agents, deployment)
    sites = dep["sites"]
    users = make_scenario(scene, seed, n_users or settings.n_users).users
    snr_all = user_snr(scene, users, settings.rho)
    snr = snr_all[sites]
    ref_sites, s_ref = sites, dep["s_ref"]
    if normalise_as is not None and normalise_as != deployment:
        other = deploy(settings, seed, n_agents, normalise_as)
        ref_sites, s_ref = other["sites"], other["s_ref"]
    r_ref = float(np.log2(1.0 + snr_all[ref_sites].sum(axis=0)).sum())
    levels = settings.levels if levels is None else levels
    n_slots = settings.n_slots if n_slots is None else n_slots
    tracks = paths(settings, seed, levels, settings.speed if speed is None else speed, n_slots)
    points = tracks.reshape(-1, 3)
    visible = los.visible(points, sites)
    feat = op.sensing_features(scene.positions[sites], points, visible)
    feat = feat.reshape(n_agents, n_slots, len(levels), op.N_FEATURES).transpose(1, 0, 2, 3)
    c_tx, c_rx = (settings.c_tx, settings.c_rx) if costs is None else costs
    return op.Episode(
        snr=snr,
        feat=np.ascontiguousarray(feat),
        tracks=tracks,
        positions=scene.positions[sites],
        lam=settings.lam if lam is None else lam,
        eps=settings.eps,
        c_tx=c_tx,
        c_rx=c_rx,
        r_ref=r_ref,
        s_ref=s_ref,
    )


def snapshot(settings: OpSettings, seed: int, n_agents: int, **kw) -> op.ModeProblem:
    """The game of slot 0 of :func:`make_episode` (a static snapshot)."""
    return make_episode(settings, seed, n_agents, n_slots=1, **kw).problem(0)


def _q(values: np.ndarray, level: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64).ravel(), level))


def _finish(result: dict, t0: float) -> dict:
    result["runtime_s"] = round(time.time() - t0, 2)
    result["machine"] = ex.machine_info()
    return result


# ------------------------------------------------------------------------------ tuning
SA_GRID = [{"t_start": 0.0, "t_end": 0.0, "cooling_fraction": 1.0}] + [
    {"t_start": t0, "t_end": t1, "cooling_fraction": frac}
    for t0 in (3e-2, 1e-2, 3e-3, 1e-3)
    for t1 in (1e-3, 1e-4, 1e-5)
    for frac in (0.5, 1.0)
    if t1 < t0
]
SA_WARM_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LLL_GRID = [
    {"tau_start": t0, "tau_end": t1, "fraction": frac}
    for t0 in (3e-2, 1e-2, 3e-3)
    for t1 in (1e-3, 1e-4)
    for frac in (0.25, 0.5)
]
COST_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
#: start values of the costs; the calibration multiplies both by one factor
COST_START = (0.004, 0.002)
TUNING_SLOTS = (0, 60, 120, 180)


def _tune_cost_task(args) -> list[dict]:
    settings, seed = args
    rows = []
    for scale in COST_GRID:
        ep = make_episode(
            settings,
            seed,
            settings.e10_agents,
            n_slots=max(TUNING_SLOTS) + 1,
            costs=(COST_START[0] * scale, COST_START[1] * scale),
        )
        for t in TUNING_SLOTS:
            problem = ep.problem(t)
            res = op.reference_solve(
                problem,
                np.random.default_rng([90, seed, t]),
                n_random=settings.reference_random_starts,
            )
            terms = problem.terms(res.profile[None, :])
            rows.append(
                {
                    "scenario": seed,
                    "scale": scale,
                    "slot": t,
                    "active": float(terms["n_tx"][0] + terms["n_rx"][0]),
                    "n_tx": float(terms["n_tx"][0]),
                    "n_rx": float(terms["n_rx"][0]),
                    "utility": res.value,
                    "rate": float(terms["rate"][0]),
                    "peb_median": float(np.median(terms["peb"][0])),
                }
            )
    return rows


def _tune_sa_task(args) -> list[dict]:
    """Cold-start annealing schedules (and log-linear learning) on one tuning snapshot.

    ``group = "e9"``: the instance type of E9 (15 users, 4 UAVs, ``D = 12``), budget
    ``e9_budget_factor * D``, random initial profiles, two repetitions.  ``group = "e11"``:
    the instance type of E11 with the scaled load, budget ``e11_sa_budget_factor * D``,
    start from all-off as in E11, two repetitions for ``D <= 24`` and one above.
    """
    settings, seed, group, n_agents = args
    if group == "e9":
        problem = snapshot(settings, seed, n_agents)
        budget, reps = settings.e9_budget_factor * n_agents, 2
    else:
        n_users, levels = e11_load(settings, n_agents, "scaled")
        problem = snapshot(settings, seed, n_agents, n_users=n_users, levels=levels)
        budget, reps = settings.e11_sa_budget_factor * n_agents, 2 if n_agents <= 24 else 1
    key = [seed, n_agents, ("e9", "e11").index(group)]
    ref = op.reference_solve(problem, np.random.default_rng([91, *key]), n_random=4)
    off = op.all_off(problem)
    results = []
    for r in range(reps):
        init = op.random_profile(problem, np.random.default_rng([92, *key, r]))
        for j, params in enumerate(SA_GRID):
            rng = np.random.default_rng([93, *key, r])
            start = init if group == "e9" else off
            res = op.simulated_annealing(problem, start, budget, rng, **params)
            results.append(("sa", j, r, res.best_value))
        if group == "e9":
            for j, params in enumerate(LLL_GRID):
                rng = np.random.default_rng([94, *key, r])
                res = op.log_linear_learning(problem, off, budget, rng, **params)
                results.append(("lll", j, r, res.best_value))
    best = max([ref.value] + [v for *_, v in results])
    name = {"sa": "sa" if group == "e9" else f"sa-e11-{n_agents}", "lll": "lll"}
    return [
        {"scenario": seed, "method": name[m], "config": j, "rep": r, "ratio": v / best}
        for m, j, r, v in results
    ]


def _tune_warm_task(args) -> list[dict]:
    """Temperature of the warm-started online annealing (zero delay, budget of ``dbr``)."""
    settings, seed = args
    ep = make_episode(settings, seed, settings.e10_agents)
    ref = op.run_reference(ep, np.random.default_rng([95, seed]), settings.reference_random_starts)
    init = ref["solutions"][0]
    dbr = op.run_distributed(ep, init, None)
    budget = max(2, int(round(dbr["evals"][1:].mean())))
    rows = []
    for j, temperature in enumerate(SA_WARM_GRID):
        out = op.run_annealing(
            ep, init, budget, np.random.default_rng([96, seed]), temperature=temperature
        )
        value = ep.evaluate(out["solutions"])["utility"]
        rows.append(
            {
                "scenario": seed,
                "method": "sa-warm",
                "config": j,
                "rep": 0,
                "ratio": float(value.sum() / ref["values"].sum()),
            }
        )
    return rows


def _select(rows: list[dict], method: str, grid: list, used) -> dict:
    """Grid point with the largest mean ratio (rounded to four decimals; ties: first)."""
    table = []
    for j, params in enumerate(grid):
        per = [
            float(
                np.mean(
                    [
                        r["ratio"]
                        for r in rows
                        if r["method"] == method and r["config"] == j and r["scenario"] == s
                    ]
                )
            )
            for s in TUNING_SEEDS
        ]
        table.append({"config": j, "parameters": params, **mean_ci(per)})
    best = max(range(len(grid)), key=lambda j: (round(table[j]["mean"], 4), -j))
    return {
        "table": table,
        "selected_config": grid[best],
        "ratio": table[best]["mean"],
        "final_settings": used,
        "final_settings_are_selected": grid[best] == used,
    }


def run_tuning(settings: OpSettings, jobs: int | None = None) -> dict:
    """Calibration of the costs and grids of the stochastic methods on the tuning scenarios."""
    t0 = time.time()
    tasks = [(settings, s) for s in TUNING_SEEDS]
    cost_rows = [r for part in ex.parallel_map(_tune_cost_task, tasks, jobs) for r in part]
    cost_table = []
    for scale in COST_GRID:
        sel = [r for r in cost_rows if r["scale"] == scale]
        row = {"scale": scale, "c_tx": COST_START[0] * scale, "c_rx": COST_START[1] * scale}
        for key in ("active", "n_tx", "n_rx", "utility", "rate", "peb_median"):
            per = [
                float(np.mean([r[key] for r in sel if r["scenario"] == s])) for s in TUNING_SEEDS
            ]
            row[key] = mean_ci(per)
        cost_table.append(row)
    target = settings.e10_agents / 2.0
    pick = min(cost_table, key=lambda r: (abs(r["active"]["mean"] - target), r["scale"]))
    costs = {
        "start": {"c_tx": COST_START[0], "c_rx": COST_START[1]},
        "target_active": target,
        "table": cost_table,
        "selected_scale": pick["scale"],
        "selected": {"c_tx": pick["c_tx"], "c_rx": pick["c_rx"]},
        "final_settings": {"c_tx": settings.c_tx, "c_rx": settings.c_rx},
        "final_settings_are_selected": bool(
            np.isclose(pick["c_tx"], settings.c_tx) and np.isclose(pick["c_rx"], settings.c_rx)
        ),
    }
    # the largest instances first, so that they are not queued behind the small ones
    sa_tasks = [
        (settings, s, "e11", n)
        for n in sorted(settings.e11_agents, reverse=True)
        for s in TUNING_SEEDS
    ]
    sa_tasks += [(settings, s, "e9", max(settings.e9_agents)) for s in TUNING_SEEDS]
    sa_rows = [r for part in ex.parallel_map(_tune_sa_task, sa_tasks, jobs) for r in part]
    warm_rows = [r for part in ex.parallel_map(_tune_warm_task, tasks, jobs) for r in part]
    selected = {
        "sa": _select(sa_rows, "sa", SA_GRID, settings.sa),
        "lll": _select(sa_rows, "lll", LLL_GRID, settings.lll),
        "sa-warm": _select(warm_rows, "sa-warm", list(SA_WARM_GRID), settings.sa_warm_temperature),
    }
    for n in settings.e11_agents:
        selected[f"sa-e11-{n}"] = _select(
            sa_rows, f"sa-e11-{n}", SA_GRID, e11_schedule(settings, n)
        )
    return _finish(
        {
            "experiment": "tuning_operation",
            "settings": settings.as_dict(),
            "tuning_seeds": list(TUNING_SEEDS),
            "costs": costs,
            "selected": selected,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E8 blockage
E8_SETS = ("trajectory", "planning")
E8_MODELS = ("blockage", "free_space")
E8_STRUCTURE = (
    ("free_space", "trajectory"),
    ("blockage", "trajectory"),
    ("blockage", "planning"),
)
E8_FUNCTIONS = ("sensing_joint", "utility", "sensing_split_tx")


def _e8_targets(settings: OpSettings, seed: int, target_set: str) -> np.ndarray:
    if target_set == "trajectory":
        return target_points(16)
    rng = np.random.default_rng([STREAM_PLANNING_POINTS, seed])
    return aerial.planning_points(get_los(), rng, settings.planning_points_per_level)


def _e8_task(args) -> list[dict]:
    settings, seed, n_agents, target_set = args
    scene, los = ex.get_scene(), get_los()
    users = make_scenario(scene, seed, settings.n_users).users
    targets = _e8_targets(settings, seed, target_set)
    scenario = Scenario(users=users, targets=targets, hotspots=SIMULATOR_HOTSPOTS)
    opts = {"lam": settings.lam, "eps": settings.eps, "rho": settings.rho}
    truth = make_problem(scene, scenario, blockage=los, **opts)
    models = {"blockage": truth, "free_space": make_problem(scene, scenario, **opts)}
    visible = los.visible(targets)
    place = placement_settings()
    budget = ex.budget_for(place, n_agents, truth.n_sites)
    key = [8, seed, n_agents, E8_SETS.index(target_set)]
    x0 = team.random_placement(np.random.default_rng([*key, 999]), truth.n_sites, n_agents)
    rand = np.argsort(np.random.default_rng([*key, 5]).random((500, truth.n_sites)), axis=1)
    random_value = float(truth.utility(rand[:, :n_agents], assume_unique=True).mean())
    stats = {}
    for j, (model, problem) in enumerate(models.items()):
        res = ex.run_method(
            "br", problem, n_agents, budget, np.random.default_rng([*key, j]), place, x0
        )
        placement = res.best_placement
        comp = truth.components(placement[None, :])
        comm, sens = truth.normalised(comp)
        peb = aerial.position_error_bound(truth.fim(placement[None, :])[0], settings.eps)
        seen = visible[placement].sum(axis=0)
        stats[model] = {
            "placement": placement,
            "utility": float(settings.lam * comm[0] + (1 - settings.lam) * sens[0]),
            "comm": float(comm[0]),
            "sens": float(sens[0]),
            "sensing_value": float(comp.logdet[0] - truth.logdet_empty),
            "rate_sum": float(comp.rate_sum[0]),
            "peb_median": float(np.median(peb)),
            "peb_p90": _q(peb, 0.9),
            "unseen": float((seen == 0).mean()),
            "single": float((seen == 1).mean()),
        }
    ref = stats["blockage"]
    rows = []
    for model, s in stats.items():
        rows.append(
            {
                "scenario": seed,
                "n_agents": n_agents,
                "target_set": target_set,
                "model": model,
                "n_points": len(targets),
                "points_no_site_sees": float((visible.sum(axis=0) == 0).mean()),
                "utility": s["utility"],
                "utility_lost": (ref["utility"] - s["utility"]) / ref["utility"],
                "gap_rate_part": settings.lam * (ref["comm"] - s["comm"]),
                "gap_sensing_part": (1 - settings.lam) * (ref["sens"] - s["sens"]),
                "sensing_value": s["sensing_value"],
                "sensing_value_lost": (ref["sensing_value"] - s["sensing_value"])
                / ref["sensing_value"],
                "rate_sum": s["rate_sum"],
                "peb_median": s["peb_median"],
                "peb_p90": s["peb_p90"],
                "unseen": s["unseen"],
                "single": s["single"],
                "overlap": len(set(s["placement"].tolist()) & set(ref["placement"].tolist()))
                / n_agents,
                "random_utility": random_value,
                "retained_gain": (s["utility"] - random_value) / (ref["utility"] - random_value),
                "placement": [int(v) for v in s["placement"]],
            }
        )
    return rows


def _e8_structure_task(args) -> list[dict]:
    """E1's sampled structure statistics for the free-space and the blockage-aware utility.

    The random streams are those of E1 (``[1, seed]`` for the fixed receive set and
    ``[1, seed, j]`` for function ``j``), so the free-space rows on the trajectory points
    repeat E1 and the other rows use the same sampled sets.
    """
    settings, seed, model, target_set = args
    scene, los = ex.get_scene(), get_los()
    users = make_scenario(scene, seed, settings.n_users).users
    scenario = Scenario(users, _e8_targets(settings, seed, target_set), SIMULATOR_HOTSPOTS)
    problem = make_problem(
        scene,
        scenario,
        lam=settings.lam,
        eps=settings.eps,
        rho=settings.rho,
        blockage=los if model == "blockage" else None,
    )
    rx = np.random.default_rng([1, seed]).choice(problem.n_sites, size=4, replace=False)
    rows = []
    for j, (name, (f, n_ground)) in enumerate(set_functions(problem, rx_sites=rx).items()):
        if name not in E8_FUNCTIONS:
            continue
        est = estimate_structure(
            f, n_ground, np.random.default_rng([1, seed, j]), n_samples=settings.e8_samples
        )
        rows.append(
            {
                "scenario": seed,
                "model": model,
                "target_set": target_set,
                "function": name,
                "eps": settings.eps,
                **est.as_dict(),
            }
        )
    return rows


def los_statistics() -> dict:
    """Statistics of the line-of-sight tables, copied from ``data/aerial_manifest.json``."""
    manifest = json.loads((default_data_dir() / aerial.LOS_MANIFEST).read_text())
    return manifest["statistics"]


def run_e8(settings: OpSettings, jobs: int | None = None) -> dict:
    """E8: placement with free-space versus blockage-aware sensing; structure of the latter."""
    t0 = time.time()
    tasks = [
        (settings, s, n, target_set)
        for target_set in E8_SETS
        for n in settings.e8_agents
        for s in range(settings.n_scenarios)
    ]
    records = [r for part in ex.parallel_map(_e8_task, tasks, jobs) for r in part]
    group = ("target_set", "n_agents", "model")
    summary = []
    for k in (
        "utility",
        "utility_lost",
        "gap_rate_part",
        "gap_sensing_part",
        "sensing_value",
        "sensing_value_lost",
        "rate_sum",
        "peb_median",
        "peb_p90",
        "unseen",
        "single",
        "overlap",
        "retained_gain",
        "points_no_site_sees",
    ):
        summary += ex.summarise(records, group, k)
    s_tasks = [
        (settings, s, model, target_set)
        for model, target_set in E8_STRUCTURE
        for s in range(settings.e8_structure_scenarios)
    ]
    rows = [r for part in ex.parallel_map(_e8_structure_task, s_tasks, jobs) for r in part]
    structure = []
    for k in ("violation_rate", "monotonicity_violation_rate"):
        structure += ex.summarise(rows, ("model", "target_set", "function"), k)
    extremes = []
    for model, target_set in E8_STRUCTURE:
        for fn in E8_FUNCTIONS:
            sel = [
                r
                for r in rows
                if r["model"] == model and r["target_set"] == target_set and r["function"] == fn
            ]
            gamma = float(np.nanmin([r["submodularity_ratio_min"] for r in sel]))
            gamma_d = float(np.nanmin([r["dr_ratio_min"] for r in sel]))
            alpha = float(np.nanmax([r["curvature_max"] for r in sel]))
            gc, gd = min(gamma, 1.0), min(gamma_d, 1.0)
            g = gc * gd**2
            extremes.append(
                {
                    "model": model,
                    "target_set": target_set,
                    "function": fn,
                    "submodularity_ratio_min": gamma,
                    "dr_ratio_min": gamma_d,
                    "curvature_max": alpha,
                    "greedy_factor": float((1 - np.exp(-alpha * gc)) / alpha)
                    if alpha > 1e-12
                    else gc,
                    "equilibrium_factor": g / (1 + g),
                }
            )
    trajectory = target_points(16)
    return _finish(
        {
            "experiment": "e8_blockage",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "los_statistics": los_statistics(),
            "trajectory_points": {
                "n": len(trajectory),
                "in_free_cells": int(get_los().is_free(trajectory).sum()),
                "seen_by_min_sites": int(get_los().visible(trajectory).sum(axis=0).min()),
            },
            "runs": records,
            "summary": summary,
            "structure_rows": rows,
            "structure": structure,
            "extremes": extremes,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E9 equilibria
E9_COSTS = ("cost", "free")
E9_METHODS = (
    "dbr-single",
    "dbr",
    "dbr-off",
    "dbr-pair-off",
    "dbr-pair",
    "lll",
    "greedy",
    "sa",
    "rs",
)


def _e9_task(args) -> dict:
    settings, seed, n_agents = args
    base = snapshot(settings, seed, n_agents)
    terms = op.exhaustive_terms(base)
    budget = settings.e9_budget_factor * n_agents
    instances, runs = [], []
    for lam in settings.e9_lams:
        for cost in E9_COSTS:
            costs = (settings.c_tx, settings.c_rx) if cost == "cost" else (0.0, 0.0)
            problem = base.with_options(lam=lam, c_tx=costs[0], c_rx=costs[1])
            values = op.utility_from_terms(problem, terms)
            best = int(np.argmax(values))
            opt = float(values[best])
            nash, stable = op.equilibrium_masks(values, n_agents)
            paired = terms["paired"]
            dead = paired[nash] == 0
            sensing = lam < 1.0  # pairing statistics are meaningless without a sensing term
            instances.append(
                {
                    "scenario": seed,
                    "n_agents": n_agents,
                    "lam": lam,
                    "cost": cost,
                    "opt": opt,
                    "opt_active": float(terms["n_tx"][best] + terms["n_rx"][best]),
                    "opt_paired_targets": int(paired[best]),
                    "n_equilibria": int(nash.sum()),
                    "n_pair_stable": int(stable.sum()),
                    "worst_ne_ratio": float(values[nash].min() / opt),
                    "mean_ne_ratio": float(values[nash].mean() / opt),
                    "worst_stable_ratio": float(values[stable].min() / opt),
                    "mean_stable_ratio": float(values[stable].mean() / opt),
                    "ne_without_pair": float(dead.mean()) if sensing else None,
                    "has_ne_without_pair": float(dead.any()) if sensing else None,
                    "ne_fewer_pairs": float((paired[nash] < paired[best]).mean())
                    if sensing
                    else None,
                    "stable_without_pair": float((paired[stable] == 0).mean()) if sensing else None,
                    "random_ratio": float(values.mean() / opt),
                }
            )
            key = [9, seed, n_agents, int(round(lam * 100)), E9_COSTS.index(cost)]
            off = op.all_off(problem)

            def record(method, rep, res, value=None, opt=opt, lam=lam, cost=cost, problem=problem):
                value = res.best_value if value is None else value
                profile = res.best_profile if value == res.best_value else res.profile
                t = problem.terms(profile[None, :])
                runs.append(
                    {
                        "scenario": seed,
                        "n_agents": n_agents,
                        "lam": lam,
                        "cost": cost,
                        "method": method,
                        "rep": rep,
                        "ratio": value / opt,
                        "is_optimal": float(value >= opt - 1e-9),
                        "n_evals": res.n_evals,
                        "no_pair": float(not t["paired"][0].any()) if lam < 1.0 else None,
                    }
                )

            record("dbr-off", 0, op.best_response(problem, off))
            record("dbr-pair-off", 0, op.pair_best_response(problem, off))
            record("greedy", 0, op.greedy(problem))
            for rep in range(settings.e9_reps):
                x0 = op.random_profile(problem, np.random.default_rng([*key, 999, rep]))

                def rng(method, rep=rep, key=key):
                    return np.random.default_rng([*key, E9_METHODS.index(method), rep])

                res = op.with_restarts(op.best_response, problem, x0, budget, rng("dbr"))
                first = res.extra["first"]
                record("dbr-single", rep, first, first.value)
                record("dbr", rep, res)
                record(
                    "dbr-pair",
                    rep,
                    op.with_restarts(op.pair_best_response, problem, x0, budget, rng("dbr-pair")),
                )
                record(
                    "lll",
                    rep,
                    op.log_linear_learning(problem, off, budget, rng("lll"), **settings.lll),
                )
                record(
                    "sa", rep, op.simulated_annealing(problem, x0, budget, rng("sa"), **settings.sa)
                )
                record("rs", rep, op.random_search(problem, budget, rng("rs")))
    return {"instances": instances, "runs": runs, "budget": budget}


def run_e9(settings: OpSettings, jobs: int | None = None) -> dict:
    """E9: all equilibria of the mode game on small deployments, and the methods' ratios."""
    t0 = time.time()
    tasks = [(settings, s, n) for n in settings.e9_agents for s in range(settings.n_scenarios)]
    parts = ex.parallel_map(_e9_task, tasks, jobs)
    instances = [r for p in parts for r in p["instances"]]
    runs = [r for p in parts for r in p["runs"]]
    summary = []
    for k in (
        "worst_ne_ratio",
        "mean_ne_ratio",
        "worst_stable_ratio",
        "mean_stable_ratio",
        "n_equilibria",
        "n_pair_stable",
        "ne_without_pair",
        "has_ne_without_pair",
        "ne_fewer_pairs",
        "stable_without_pair",
        "opt_active",
        "random_ratio",
    ):
        summary += ex.summarise(instances, ("lam", "cost", "n_agents"), k)
        summary += [
            {**r, "n_agents": "all"}
            for r in ex.summarise(
                [{**x, "unit": (x["scenario"], x["n_agents"])} for x in instances],
                ("lam", "cost"),
                k,
                unit="unit",
            )
        ]
    for k in ("ratio", "is_optimal", "n_evals", "no_pair"):
        summary += ex.summarise(runs, ("lam", "cost", "n_agents", "method"), k)
        summary += [
            {**r, "n_agents": "all"}
            for r in ex.summarise(
                [{**x, "unit": (x["scenario"], x["n_agents"])} for x in runs],
                ("lam", "cost", "method"),
                k,
                unit="unit",
            )
        ]
    minima = []
    for lam in settings.e9_lams:
        for cost in E9_COSTS:
            inst = [r for r in instances if r["lam"] == lam and r["cost"] == cost]
            for name, field_ in (
                ("worst-equilibrium", "worst_ne_ratio"),
                ("worst-pair-stable", "worst_stable_ratio"),
            ):
                minima.append(
                    {
                        "lam": lam,
                        "cost": cost,
                        "method": name,
                        "min_ratio": min(r[field_] for r in inst),
                    }
                )
            for m in E9_METHODS:
                sel = [
                    r["ratio"]
                    for r in runs
                    if r["lam"] == lam and r["cost"] == cost and r["method"] == m
                ]
                minima.append({"lam": lam, "cost": cost, "method": m, "min_ratio": min(sel)})
    return _finish(
        {
            "experiment": "e9_mode_equilibria",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "budgets": {str(n): settings.e9_budget_factor * n for n in settings.e9_agents},
            "instances": instances,
            "runs": runs,
            "summary": summary,
            "minima": minima,
            "cost_sweep": run_cost_sweep(settings, jobs),
        },
        t0,
    )


# ------------------------------------------------------------------------------ E10 online
def _controller_row(
    ep: op.Episode, profiles, evals, messages, best_known, static, subrounds=None, **labels
) -> dict:
    terms = ep.evaluate(profiles)
    paired = terms["paired"]
    total, gap = float(terms["utility"].sum()), float(best_known.sum() - static)
    return {
        **labels,
        "relative_utility": float(total / best_known.sum()),
        # share of the gap between the static assignment (0) and the best known (1)
        "gain_captured": (total - static) / gap if gap > 0.0 else None,
        "subrounds": None if subrounds is None else float(np.asarray(subrounds)[1:].mean()),
        "utility": float(terms["utility"].mean()),
        "rate": float(terms["rate"].mean()),
        "peb_median": float(np.median(terms["peb"])),
        "peb_p90": _q(terms["peb"], 0.9),
        "unpaired": float(1.0 - paired.mean()),
        "active": float((terms["n_tx"] + terms["n_rx"]).mean()),
        "n_tx": float(terms["n_tx"].mean()),
        "n_rx": float(terms["n_rx"].mean()),
        "switches": float(terms["switches"][1:].mean()),
        "evals": float(np.asarray(evals)[1:].mean()),
        "messages": float(np.asarray(messages)[1:].mean()),
    }


def _e10_task(args) -> dict:
    settings, seed, speed, deployment = args
    n_agents = settings.e10_agents
    ep = make_episode(
        settings, seed, n_agents, speed=speed, deployment=deployment, normalise_as="planned"
    )
    key = [10, seed, int(round(speed * 10)), ("planned", "random").index(deployment)]
    ref = op.run_reference(ep, np.random.default_rng([*key, 0]), settings.reference_random_starts)
    init = ref["solutions"][0]
    zeros = np.zeros(ep.n_slots)
    # name -> (applied profiles, evaluations per slot, messages per slot, labels)
    runs: dict[str, tuple] = {}

    def add(name, profiles, evals, messages, subrounds=None, **labels):
        runs[name] = (profiles, evals, messages, subrounds, {"controller": name, **labels})

    add(
        "reference",
        ref["solutions"],
        ref["evals"],
        op.command_messages(ref["solutions"]),
        family="reference",
    )
    add("static", np.repeat(init[None, :], ep.n_slots, axis=0), zeros, zeros, family="static")
    distributed = {}
    full = deployment == "planned"
    for r in (*settings.e10_subrounds, None) if full else (None,):
        out = op.run_distributed(ep, init, r)
        name = "dbr-conv" if r is None else f"dbr-r{r}"
        distributed[name] = out
        add(
            name,
            out["profiles"],
            out["evals"],
            out["messages"],
            out["subrounds"],
            family="distributed",
            limit=-1 if r is None else r,
        )
    if full:
        for size in settings.e10_clusters:
            for r in (1, 2, None):
                out = op.run_distributed(ep, init, r, cluster=size)
                name = f"dbr-L{size}-" + ("conv" if r is None else f"r{r}")
                add(
                    name,
                    out["profiles"],
                    out["evals"],
                    out["messages"],
                    out["subrounds"],
                    family="clustered",
                    limit=-1 if r is None else r,
                    cluster=size,
                )
        out = op.run_distributed(ep, init, pair=True)
        add("dbr-pair", out["profiles"], out["evals"], out["messages"], family="distributed-pair")
        hold = (np.arange(ep.n_slots) // settings.e10_period) * settings.e10_period
        periodic = ref["solutions"][hold]
        evals = np.where(np.arange(ep.n_slots) == hold, ref["evals"], 0.0)
        add("periodic", periodic, evals, op.command_messages(periodic), family="periodic")
        greedy = op.run_greedy(ep, init)
        for d in settings.e10_delays:
            profiles = op.delayed(greedy["solutions"], d)
            add(
                f"greedy-d{d}",
                profiles,
                greedy["evals"],
                op.command_messages(profiles),
                family="greedy",
                delay=d,
            )
        for j, (name, out) in enumerate(distributed.items()):
            budget = max(2, int(round(out["evals"][1:].mean())))
            sa = op.run_annealing(
                ep,
                init,
                budget,
                np.random.default_rng([*key, 1, j]),
                temperature=settings.sa_warm_temperature,
            )
            for d in settings.e10_delays:
                profiles = op.delayed(sa["solutions"], d)
                add(
                    f"sa@{name}-d{d}",
                    profiles,
                    sa["evals"],
                    op.command_messages(profiles),
                    family="sa",
                    delay=d,
                    matched=name,
                    budget=budget,
                )
    best_known = ref["values"].copy()
    for profiles, *_ in runs.values():
        best_known = np.maximum(best_known, ep.evaluate(profiles)["utility"])
    static = float(ep.evaluate(runs["static"][0])["utility"].sum())
    rows = []
    for profiles, evals, messages, subrounds, labels in runs.values():
        rows.append(
            _controller_row(
                ep,
                profiles,
                evals,
                messages,
                best_known,
                static,
                subrounds,
                scenario=seed,
                speed=speed,
                deployment=deployment,
                **labels,
            )
        )
    colours = distributed["dbr-conv"]["colours"][1:]
    graph = {
        "scenario": seed,
        "speed": speed,
        "deployment": deployment,
        "colours": float(colours.mean()),
        "visible_share": float((ep.feat[..., 0] > 0).mean()),
        "best_known_utility": float(best_known.mean()),
        "reference_evals": float(ref["evals"].mean()),
    }
    return {"rows": rows, "graph": graph}


E10_METRICS = (
    "relative_utility",
    "gain_captured",
    "subrounds",
    "utility",
    "rate",
    "peb_median",
    "peb_p90",
    "unpaired",
    "active",
    "n_tx",
    "n_rx",
    "switches",
    "evals",
    "messages",
)


def run_e10(settings: OpSettings, jobs: int | None = None) -> dict:
    """E10: online tracking of moving UAVs; distributed versus delayed central control."""
    t0 = time.time()
    tasks = [
        (settings, s, v, "planned")
        for v in settings.e10_speeds
        for s in range(settings.n_scenarios)
    ]
    tasks += [(settings, s, settings.speed, "random") for s in range(settings.n_scenarios)]
    parts = ex.parallel_map(_e10_task, tasks, jobs)
    records = [r for p in parts for r in p["rows"]]
    graphs = [p["graph"] for p in parts]
    group = ("deployment", "speed", "controller")
    summary = []
    for k in E10_METRICS:
        summary += ex.summarise(records, group, k)
    for k in ("colours", "visible_share", "best_known_utility", "reference_evals"):
        summary += [
            {**r, "controller": "graph"} for r in ex.summarise(graphs, ("deployment", "speed"), k)
        ]
    # Paired comparison: annealing with the budget of a distributed controller and a decision
    # delay, minus that distributed controller.
    index = {
        (r["deployment"], r["speed"], r["controller"], r["scenario"]): r["relative_utility"]
        for r in records
    }
    names = [f"dbr-r{r}" for r in settings.e10_subrounds] + ["dbr-conv"]
    names += [f"dbr-L{size}-{r}" for size in settings.e10_clusters for r in ("r1", "r2", "conv")]
    differences, crossover = [], []
    for speed in settings.e10_speeds:
        for name in names:
            for budget in ("matched", "full"):
                central = name if budget == "matched" else "dbr-conv"
                if ("planned", speed, f"sa@{central}-d0", 0) not in index:
                    continue  # the clustered controllers have no budget-matched annealing
                first, first_significant, last_ahead = None, None, None
                for d in settings.e10_delays:
                    diff = [
                        index[("planned", speed, f"sa@{central}-d{d}", s)]
                        - index[("planned", speed, name, s)]
                        for s in range(settings.n_scenarios)
                    ]
                    ci = mean_ci(diff)
                    differences.append(
                        {"speed": speed, "controller": name, "budget": budget, "delay": d, **ci}
                    )
                    if first is None and ci["mean"] < 0.0:
                        first = d
                    if first_significant is None and ci["hi"] < 0.0:
                        first_significant = d
                    if ci["lo"] > 0.0:
                        last_ahead = d
                crossover.append(
                    {
                        "speed": speed,
                        "controller": name,
                        "budget": budget,
                        "first_delay_behind": first,
                        "first_delay_significantly_behind": first_significant,
                        "last_delay_significantly_ahead": last_ahead,
                    }
                )
    return _finish(
        {
            "experiment": "e10_online_tracking",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "runs": records,
            "graphs": graphs,
            "summary": summary,
            "differences": differences,
            "crossover": crossover,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E11 locality
E11_LOADS = ("scaled", "fixed")


def e11_schedule(settings: OpSettings, n_agents: int) -> dict:
    """Cold annealing schedule used in E11 for a deployment of ``n_agents`` APs."""
    return settings.sa_e11.get(str(n_agents), settings.sa)


def e11_load(settings: OpSettings, n_agents: int, load: str) -> tuple[int, tuple[int, ...]]:
    """Users and flight levels of the UAVs for a deployment of ``n_agents`` APs.

    ``fixed``: the default 15 users and 4 UAVs.  ``scaled``: the same ratio as for 24 APs,
    i.e. ``15 D / 24`` users (rounded) and ``D / 6`` UAVs, the levels repeating in order.
    """
    if load == "fixed":
        return settings.n_users, settings.levels
    n_users = max(1, int(round(settings.n_users * n_agents / 24)))
    n_targets = max(1, int(round(len(settings.levels) * n_agents / 24)))
    return n_users, tuple(settings.levels[q % len(settings.levels)] for q in range(n_targets))


def _first_reach(trace: list[tuple[int, float]], level: float):
    hit = [e for e, v in trace if v >= level]
    return hit[0] if hit else None


def _e11_task(args) -> dict:
    settings, seed, n_agents, load = args
    n_users, levels = e11_load(settings, n_agents, load)
    problem = snapshot(settings, seed, n_agents, n_users=n_users, levels=levels)
    key = [11, seed, n_agents, n_users, len(levels)]
    off = op.all_off(problem)
    degree = problem.degrees()
    colours = op.greedy_colouring(problem.interaction_graph())
    dbr = op.best_response(problem, off)
    col = op.coloured_best_response(problem, off, colours=colours)
    greedy = op.greedy(problem)
    budget = settings.e11_sa_budget_factor * n_agents
    schedule = e11_schedule(settings, n_agents)
    sa = op.simulated_annealing(problem, off, budget, np.random.default_rng([*key, 1]), **schedule)
    ref = op.reference_solve(
        problem,
        np.random.default_rng([*key, 0]),
        [dbr.profile, greedy.profile, sa.best_profile],
        settings.reference_random_starts,
    )
    best = max(ref.value, dbr.value, col.value, greedy.value, sa.best_value)
    n_terms = problem.n_users + problem.n_targets
    row = {
        "scenario": seed,
        "n_agents": n_agents,
        "load": load,
        "n_users": n_users,
        "n_targets": len(levels),
        "cluster": 0,
        "degree_mean": float(degree.mean()),
        "degree_min": float(degree.min()),
        "degree_max": float(degree.max()),
        "density": float(degree.mean() / (n_agents - 1)),
        "colours": int(colours.max()) + 1,
        "clique": op.clique_lower_bound(problem),
        "local_terms": float((problem.n_reach + problem.n_sees).mean()),
        "local_share": float((problem.n_reach + problem.n_sees).mean() / n_terms),
        "best_known": best,
        "dbr_ratio": dbr.value / best,
        "dbr_moves": dbr.n_moves,
        "dbr_evals": dbr.n_evals,
        "dbr_evals_99": _first_reach(dbr.trace, 0.99 * best),
        "dbr_work_per_step": dbr.work / max(dbr.n_steps, 1),
        "global_work_per_step": 3 * n_terms,
        "colour_ratio": col.value / best,
        "colour_subrounds": col.n_subrounds,
        "colour_moves": col.n_moves,
        "colour_rounds": col.n_subrounds / (int(colours.max()) + 1),
        "greedy_ratio": greedy.value / best,
        "greedy_evals": greedy.n_evals,
        "greedy_evals_99": _first_reach(greedy.trace, 0.99 * best),
        "sa_ratio": sa.best_value / best,
        "sa_evals": sa.n_evals,
        "sa_evals_99": _first_reach(sa.trace, 0.99 * best),
        "reference_ratio": ref.value / best,
        "reference_evals": ref.n_evals,
        "active": float((ref.profile != op.OFF).sum()),
    }
    rows = [row]
    for size in settings.e11_clusters:
        local = problem.clustered(size)
        c = op.greedy_colouring(local.interaction_graph())
        res = op.coloured_best_response(local, off, colours=c)
        full = float(problem.utility(res.profile[None, :])[0])
        polished = op.best_response(problem, res.profile)
        d = local.degrees()
        rows.append(
            {
                "scenario": seed,
                "n_agents": n_agents,
                "load": load,
                "n_users": n_users,
                "n_targets": len(levels),
                "cluster": size,
                "degree_mean": float(d.mean()),
                "degree_min": float(d.min()),
                "degree_max": float(d.max()),
                "density": float(d.mean() / (n_agents - 1)),
                "colours": int(c.max()) + 1,
                "clique": op.clique_lower_bound(local),
                "local_terms": float((local.n_reach + local.n_sees).mean()),
                "local_share": float((local.n_reach + local.n_sees).mean() / n_terms),
                "best_known": best,
                "colour_ratio": full / best,
                "colour_subrounds": res.n_subrounds,
                "colour_moves": res.n_moves,
                "colour_rounds": res.n_subrounds / (int(c.max()) + 1),
                "colour_evals": res.n_evals,
                "polished_ratio": polished.value / best,
                "polish_moves": polished.n_moves,
                "active": float((res.profile != op.OFF).sum()),
            }
        )
    return {"rows": rows}


def run_e11(settings: OpSettings, jobs: int | None = None) -> dict:
    """E11: interaction graph, colours and convergence versus the size of the deployment."""
    t0 = time.time()
    tasks = [
        (settings, s, n, load)
        for load in E11_LOADS
        for n in settings.e11_agents
        for s in range(settings.n_scenarios)
    ]
    records = [r for p in ex.parallel_map(_e11_task, tasks, jobs) for r in p["rows"]]
    keys = sorted({k for r in records for k in r} - {"scenario", "n_agents", "load", "cluster"})
    summary = []
    for k in keys:
        summary += ex.summarise(records, ("load", "n_agents", "cluster"), k)
    for k in ("dbr_evals_99", "greedy_evals_99", "sa_evals_99"):
        for r in records:
            if r["cluster"] == 0:
                r[k.replace("evals_99", "reached_99")] = float(r[k] is not None)
    for k in ("dbr_reached_99", "greedy_reached_99", "sa_reached_99"):
        summary += ex.summarise(records, ("load", "n_agents", "cluster"), k)
    return _finish(
        {
            "experiment": "e11_locality",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "runs": records,
            "summary": summary,
        },
        t0,
    )


# ------------------------------------------------------------------------------ E12 failures
E12_REPAIRS = (
    "before",
    "none",
    "hop1",
    "hop2",
    "cluster4",
    "cluster8",
    "dbr",
    "sa",
    "greedy",
)


def _neighbourhood(adj: np.ndarray, seeds: np.ndarray, hops: int) -> np.ndarray:
    mask = np.zeros(len(adj), dtype=bool)
    mask[seeds] = True
    for _ in range(hops):
        mask |= adj[mask].any(axis=0)
    return mask


def _e12_task(args) -> list[dict]:
    settings, seed, n_failed, rep = args
    n_agents = settings.e12_agents
    problem = snapshot(settings, seed, n_agents)
    key = [12, seed, n_failed, rep]
    before = op.reference_solve(
        problem, np.random.default_rng([12, seed, 0]), n_random=settings.reference_random_starts
    )
    active = np.flatnonzero(before.profile != op.OFF)
    rng = np.random.default_rng([*key, 1])
    failed = np.sort(rng.choice(active, size=min(n_failed, len(active)), replace=False))
    after = problem.without(failed)
    start = before.profile.copy()
    start[failed] = op.OFF
    alive = after.free
    adj = problem.interaction_graph()

    def restricted(mask):
        return np.flatnonzero(mask & alive)

    sets = {
        "hop1": restricted(_neighbourhood(adj, failed, 1)),
        "hop2": restricted(_neighbourhood(adj, failed, 2)),
        "cluster4": restricted(_neighbourhood(problem.clustered(4).interaction_graph(), failed, 1)),
        "cluster8": restricted(_neighbourhood(problem.clustered(8).interaction_graph(), failed, 1)),
    }
    results = {"none": (start, 0, 0, 0)}
    for name, agents in sets.items():
        res = op.best_response(after, start, agents=agents)
        results[name] = (res.profile, res.n_evals, res.messages, len(agents))
    dbr = op.best_response(after, start)
    results["dbr"] = (dbr.profile, dbr.n_evals, dbr.messages, int(alive.sum()))
    budget = max(dbr.n_evals, 2)
    sa = op.simulated_annealing(
        after,
        start,
        budget,
        np.random.default_rng([*key, 2]),
        t_start=settings.sa_warm_temperature,
        t_end=settings.sa_warm_temperature,
    )
    results["sa"] = (sa.best_profile, sa.n_evals, None, int(alive.sum()))
    greedy = op.greedy(after)
    results["greedy"] = (greedy.profile, greedy.n_evals, None, int(alive.sum()))
    ref = op.reference_solve(
        after,
        np.random.default_rng([*key, 3]),
        [start, dbr.profile],
        settings.reference_random_starts,
    )
    best = max([ref.value] + [float(after.utility(p[None, :])[0]) for p, *_ in results.values()])
    terms = problem.terms(before.profile[None, :])
    rows = [
        {
            "scenario": seed,
            "n_failed": n_failed,
            "rep": rep,
            "repair": "before",
            "utility": before.value,
            "ratio_to_before": 1.0,
            "peb_median": float(np.median(terms["peb"][0])),
            "peb_max": float(terms["peb"][0].max()),
            "unpaired": float(1.0 - terms["paired"][0].mean()),
            "rate": float(terms["rate"][0]),
            "active": float(len(active)),
        }
    ]
    for name, (profile, evals, messages, allowed) in results.items():
        terms = after.terms(profile[None, :])
        changed = int((profile != start).sum())
        rows.append(
            {
                "scenario": seed,
                "n_failed": n_failed,
                "rep": rep,
                "repair": name,
                "utility": float(terms["utility"][0]),
                "ratio": float(terms["utility"][0] / best),
                "ratio_to_before": float(terms["utility"][0] / before.value),
                "peb_median": float(np.median(terms["peb"][0])),
                "peb_max": float(terms["peb"][0].max()),
                "unpaired": float(1.0 - terms["paired"][0].mean()),
                "rate": float(terms["rate"][0]),
                "active": float((profile != op.OFF).sum()),
                "changed": changed,
                "evals": evals,
                "messages": changed if messages is None else messages,
                "allowed": allowed,
                "degree_failed": float(adj[failed].sum(axis=1).mean()),
            }
        )
    return rows


def run_e12(settings: OpSettings, jobs: int | None = None) -> dict:
    """E12: failures of active APs; local repair, full best response and central re-solve."""
    t0 = time.time()
    tasks = [
        (settings, s, f, rep)
        for f in settings.e12_failures
        for s in range(settings.n_scenarios)
        for rep in range(settings.e12_reps)
    ]
    records = [r for part in ex.parallel_map(_e12_task, tasks, jobs) for r in part]
    summary = []
    for k in (
        "ratio",
        "ratio_to_before",
        "utility",
        "peb_median",
        "peb_max",
        "unpaired",
        "rate",
        "active",
        "changed",
        "evals",
        "messages",
        "allowed",
        "degree_failed",
    ):
        summary += ex.summarise(records, ("n_failed", "repair"), k)
    return _finish(
        {
            "experiment": "e12_failures",
            "settings": settings.as_dict(),
            "scenario_seeds": list(range(settings.n_scenarios)),
            "runs": records,
            "summary": summary,
        },
        t0,
    )


# ------------------------------------------------------------------------------ cost sweep
def _cost_task(args) -> list[dict]:
    settings, seed, scale = args
    ep = make_episode(
        settings,
        seed,
        settings.e10_agents,
        n_slots=max(TUNING_SLOTS) + 1,
        costs=(settings.c_tx * scale, settings.c_rx * scale),
    )
    rows = []
    for t in TUNING_SLOTS:
        problem = ep.problem(t)
        ref = op.reference_solve(
            problem, np.random.default_rng([13, seed, t]), n_random=settings.reference_random_starts
        )
        dbr = op.best_response(problem, op.all_off(problem))
        terms = problem.terms(ref.profile[None, :])
        rows.append(
            {
                "scenario": seed,
                "scale": scale,
                "slot": t,
                "active": float(terms["n_tx"][0] + terms["n_rx"][0]),
                "n_tx": float(terms["n_tx"][0]),
                "n_rx": float(terms["n_rx"][0]),
                "utility": ref.value,
                "rate": float(terms["rate"][0]),
                "peb_median": float(np.median(terms["peb"][0])),
                "unpaired": float(1.0 - terms["paired"][0].mean()),
                "dbr_ratio": dbr.value / max(ref.value, dbr.value),
            }
        )
    return rows


def run_cost_sweep(settings: OpSettings, jobs: int | None = None) -> list[dict]:
    """Sensitivity of the equilibrium to the cost scale (part of the E9 result file)."""
    tasks = [
        (settings, s, scale) for scale in settings.cost_scales for s in range(settings.n_scenarios)
    ]
    rows = [r for part in ex.parallel_map(_cost_task, tasks, jobs) for r in part]
    summary = []
    for k in ("active", "n_tx", "n_rx", "utility", "rate", "peb_median", "unpaired", "dbr_ratio"):
        summary += ex.summarise(rows, ("scale",), k)
    return summary


EXPERIMENTS = {
    "e8": run_e8,
    "e9": run_e9,
    "e10": run_e10,
    "e11": run_e11,
    "e12": run_e12,
}
