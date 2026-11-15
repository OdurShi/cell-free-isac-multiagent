"""Centralised optimisers of the same objective: greedy, simulated annealing, a genetic
algorithm, random search and exhaustive search.

Simulated annealing and the genetic algorithm follow ``simulation/compare_baselines.py`` of
the simulator (single-site swap proposals with Metropolis acceptance; rank selection,
one-point crossover, repair, mutation and one elite), with the temperature schedule and the
evaluation budget as parameters.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from cfisac_ma.objective import Evaluator, Problem
from cfisac_ma.team import RunResult, _result, random_placement


def greedy(ev: Evaluator, n_agents: int) -> RunResult:
    """Greedy selection: add, ``n_agents`` times, the site with the largest utility gain."""
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    placement = np.full(n_agents, -1, dtype=np.int64)
    free = np.ones(n_sites, dtype=bool)
    value = np.nan
    for i in range(n_agents):
        cand = np.flatnonzero(free)
        if ev.remaining < len(cand):
            break
        values = ev.replace(placement, i, cand)
        best = int(np.argmax(values))
        placement[i] = cand[best]
        free[cand[best]] = False
        value = float(values[best])
    return _result("greedy", ev, placement, value, start, n_moves=n_agents, converged=True)


def random_search(
    ev: Evaluator, n_agents: int, rng: np.random.Generator, batch: int = 256
) -> RunResult:
    """Uniformly random placements of distinct sites until the budget is used."""
    if ev.budget is None:
        raise ValueError("random_search needs an evaluator with a budget")
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    while ev.remaining >= 1:
        b = int(min(batch, ev.remaining))
        placements = np.argsort(rng.random((b, n_sites)), axis=1)[:, :n_agents]
        ev.utility(placements)
    return _result("random-search", ev, ev.best_placement, ev.best_value, start)


def simulated_annealing(
    ev: Evaluator,
    init: np.ndarray,
    rng: np.random.Generator,
    *,
    t_start: float = 100.0,
    t_end: float | None = None,
    cooling: float | None = 0.99,
    cooling_fraction: float = 1.0,
) -> RunResult:
    """Simulated annealing with single-site swap proposals.

    A proposal moves one randomly chosen access point to a random unoccupied site; it is
    accepted if it improves the utility, otherwise with probability
    ``exp((U_new - U_old) / T)``.  The temperature decays geometrically, either by the
    factor ``cooling`` per proposal (simulator: ``t_start=100``, ``cooling=0.99``) or, if
    ``t_end`` is given, from ``t_start`` to ``t_end`` over the first ``cooling_fraction`` of
    the budget, after which it stays at ``t_end``.
    """
    if ev.budget is None:
        raise ValueError("simulated_annealing needs an evaluator with a budget")
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    placement = np.array(init)
    n_agents = len(placement)
    value = float(ev.utility(placement[None, :])[0])
    steps = int(ev.remaining)
    floor = 1e-300
    if t_end is not None:
        cooling_steps = max(int(round(cooling_fraction * steps)), 2)
        cooling = (t_end / t_start) ** (1.0 / (cooling_steps - 1))
        floor = t_end
    temp = t_start
    occupied = np.zeros(n_sites, dtype=bool)
    occupied[placement] = True
    moves = 0
    # Random numbers for all proposals are drawn up front (same stream for any schedule).
    agents = rng.integers(n_agents, size=steps)
    accept_u = rng.random(steps)
    for s in range(steps):
        i = agents[s]
        new_site = int(rng.integers(n_sites))
        while occupied[new_site]:
            new_site = int(rng.integers(n_sites))
        cand = placement.copy()
        cand[i] = new_site
        new_value = float(ev.utility(cand[None, :])[0])
        delta = new_value - value
        if delta > 0 or accept_u[s] < math.exp(delta / temp):
            occupied[placement[i]] = False
            occupied[new_site] = True
            placement, value = cand, new_value
            moves += 1
        temp = max(temp * cooling, floor)
    return _result("sa", ev, placement, value, start, n_moves=moves)


def genetic_algorithm(
    ev: Evaluator,
    n_agents: int,
    rng: np.random.Generator,
    *,
    population: int = 50,
    mutation_rate: float = 0.1,
    init: np.ndarray | None = None,
) -> RunResult:
    """Genetic algorithm of the simulator's baseline script.

    Rank-proportional selection, one-point crossover of the site lists, repair of repeated
    sites with random unused ones, one mutated site with probability ``mutation_rate`` and
    one elite (the best individual so far).  ``init``, if given, replaces the first
    individual of the initial population (warm start).
    """
    if ev.budget is None:
        raise ValueError("genetic_algorithm needs an evaluator with a budget")
    n_sites = ev.problem.n_sites
    start = ev.n_evals
    pop = np.stack([random_placement(rng, n_sites, n_agents) for _ in range(population)])
    if init is not None:
        pop[0] = init
    ranks_p = np.arange(1, population + 1, dtype=np.float64)
    ranks_p /= ranks_p.sum()
    generations = 0
    best_value, best = -np.inf, pop[0]
    while ev.remaining >= 1:
        if ev.remaining < len(pop):
            pop = pop[: int(ev.remaining)]
        fitness = ev.utility(pop)
        generations += 1
        j = int(np.argmax(fitness))
        if fitness[j] > best_value:
            best_value, best = float(fitness[j]), pop[j].copy()
        if len(pop) < population:
            break
        order = np.argsort(fitness, kind="stable")  # worst first: rank 1 ... population
        parents = order[rng.choice(population, size=(population - 1, 2), p=ranks_p)]
        cuts = rng.integers(1, n_agents, size=population - 1) if n_agents > 1 else None
        children = [best]
        for c in range(population - 1):
            p1, p2 = pop[parents[c, 0]], pop[parents[c, 1]]
            cut = int(cuts[c]) if cuts is not None else 0
            child = list(dict.fromkeys(np.concatenate([p1[:cut], p2[cut:]]).tolist()))
            used = set(child)
            while len(child) < n_agents:
                s = int(rng.integers(n_sites))
                if s not in used:
                    child.append(s)
                    used.add(s)
            if rng.random() < mutation_rate:
                s = int(rng.integers(n_sites))
                while s in used:
                    s = int(rng.integers(n_sites))
                child[int(rng.integers(n_agents))] = s
            children.append(np.array(child))
        pop = np.stack(children)
    return _result("ga", ev, best, best_value, start, n_rounds=generations)


def all_subsets(n_sites: int, size: int) -> np.ndarray:
    """All ``size``-subsets of ``range(n_sites)`` in lexicographic order, ``[C, size]``."""
    count = math.comb(n_sites, size)
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n_sites), size)),
        dtype=np.int64,
        count=count * size,
    )
    return flat.reshape(count, size)


def exhaustive_values(
    problem: Problem, size: int, chunk: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    """Utility of every ``size``-subset of the problem's sites.

    Returns ``(subsets [C, size], values [C])``.  Not charged to any evaluator: this is the
    ground truth for small instances.
    """
    subsets = all_subsets(problem.n_sites, size)
    values = np.empty(len(subsets))
    for lo in range(0, len(subsets), chunk):
        values[lo : lo + chunk] = problem.utility(subsets[lo : lo + chunk], assume_unique=True)
    return subsets, values


def exhaustive_terms(
    problem: Problem, size: int, chunk: int = 100_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised communication and sensing terms of every ``size``-subset.

    Returns ``(subsets [C, size], comm [C], sens [C])``; the utility for a weight ``lam`` is
    ``lam * comm + (1 - lam) * sens``, exactly as :meth:`Problem.utility` computes it.
    """
    subsets = all_subsets(problem.n_sites, size)
    comm = np.empty(len(subsets))
    sens = np.empty(len(subsets))
    for lo in range(0, len(subsets), chunk):
        part = problem.components(subsets[lo : lo + chunk], assume_unique=True)
        comm[lo : lo + chunk], sens[lo : lo + chunk] = problem.normalised(part)
    return subsets, comm, sens


def subset_rank(subsets: np.ndarray, n_sites: int) -> np.ndarray:
    """Lexicographic rank of sorted subsets ``[C, size]`` among all subsets of that size."""
    size = subsets.shape[1]
    # Number of subsets that precede: for each position j, the subsets that agree on the
    # first j elements and have a smaller element at position j.
    table = np.zeros((n_sites + 1, size + 1), dtype=np.int64)
    for n in range(n_sites + 1):
        for k in range(size + 1):
            table[n, k] = math.comb(n, k)
    total = table[n_sites, size]
    # rank = C(n, size) - 1 - sum_j C(n - 1 - s_j, size - j)
    rev = np.zeros(len(subsets), dtype=np.int64)
    for j in range(size):
        rev += table[n_sites - 1 - subsets[:, j], size - j]
    return total - 1 - rev


def nash_equilibria(
    subsets: np.ndarray, values: np.ndarray, n_sites: int, tol: float = 1e-12
) -> np.ndarray:
    """Boolean mask of the subsets that are pure Nash equilibria of the team game.

    A placement on ``size`` distinct sites is an equilibrium iff no agent can raise ``U`` by
    moving to an unoccupied site (for a monotone ``U`` moving onto an occupied site never
    helps), i.e. iff the subset is optimal within its single-swap neighbourhood.

    ``values`` may be ``[C]`` or ``[L, C]`` (several utilities on the same subsets, e.g. one
    per weight ``lam``); the mask has the same shape.
    """
    size = subsets.shape[1]
    values = np.asarray(values)
    stacked = np.atleast_2d(values)
    is_ne = np.ones(stacked.shape, dtype=bool)
    for j in range(size):
        for c in range(n_sites):
            neighbour = subsets.copy()
            neighbour[:, j] = c
            valid = (subsets != c).all(axis=1)
            rank = subset_rank(np.sort(neighbour[valid], axis=1), n_sites)
            better = stacked[:, rank] > stacked[:, valid] + tol
            is_ne[:, valid] &= ~better
    return is_ne[0] if values.ndim == 1 else is_ne
