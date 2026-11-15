"""Optimisers: budgets, validity of the returned placements, simple exact cases."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from conftest import random_problem

from cfisac_ma import centralized as cen
from cfisac_ma import experiments as ex
from cfisac_ma import team
from cfisac_ma.objective import Evaluator, Problem


def test_all_subsets_and_rank():
    subsets = cen.all_subsets(7, 3)
    assert len(subsets) == math.comb(7, 3)
    assert [tuple(s) for s in subsets] == list(itertools.combinations(range(7), 3))
    assert np.array_equal(cen.subset_rank(subsets, 7), np.arange(len(subsets)))


def test_exhaustive_values_match_utility():
    p = random_problem(0, n_sites=9)
    subsets, values = cen.exhaustive_values(p, 4, chunk=17)
    assert np.allclose(values, p.utility(subsets))
    same, comm, sens = cen.exhaustive_terms(p, 4, chunk=17)
    assert np.array_equal(same, subsets)
    for lam in (0.0, 0.3, 1.0):
        q = p.with_options(lam=lam)
        assert np.array_equal(lam * comm + (1.0 - lam) * sens, q.utility(subsets))


def test_nash_equilibria_for_several_utilities_at_once():
    p = random_problem(3, n_sites=10)
    subsets, comm, sens = cen.exhaustive_terms(p, 3)
    stacked = np.stack([lam * comm + (1.0 - lam) * sens for lam in (1.0, 0.5, 0.0)])
    masks = cen.nash_equilibria(subsets, stacked, 10)
    assert masks.shape == stacked.shape
    for row, mask in zip(stacked, masks, strict=True):
        assert np.array_equal(mask, cen.nash_equilibria(subsets, row, 10))
        assert mask[int(np.argmax(row))]


@pytest.mark.parametrize("name", ex.METHODS)
def test_methods_respect_the_budget_and_return_valid_placements(name):
    p = random_problem(1, n_sites=20)
    settings = ex.PASSES[ex.FINAL_PASS]
    budget = 400
    init = np.array([0, 5, 9, 13])
    res = ex.run_method(name, p, 4, budget, np.random.default_rng(3), settings, init)
    assert 0 < res.n_evals <= budget
    assert res.best_placement.shape == (4,)
    assert ((res.best_placement >= 0) & (res.best_placement < 20)).all()
    assert p.utility(res.best_placement[None, :])[0] == pytest.approx(res.best_value, rel=1e-12)
    assert res.trace[-1][1] == pytest.approx(res.best_value)
    assert res.trace[-1][0] <= budget


def test_greedy_is_optimal_for_a_modular_objective():
    # one user per site, far apart targets irrelevant (lam = 1): R is a sum of concave
    # functions of disjoint terms, i.e. modular, so greedy picks the largest terms
    snr = np.diag([5.0, 50.0, 1.0, 500.0, 20.0, 2.0])
    w = np.ones((6, 1))
    u = np.tile(np.array([1.0, 0.0, 0.0]), (6, 1, 1))
    p = Problem(snr, w, u, lam=1.0)
    res = cen.greedy(Evaluator(p), 3)
    assert set(res.placement.tolist()) == {1, 3, 4}
    _, values = cen.exhaustive_values(p, 3)
    assert res.value == pytest.approx(values.max())
    assert res.n_evals == 6 + 5 + 4


def test_best_response_reaches_a_swap_optimal_set():
    p = random_problem(2, n_sites=14)
    res = team.best_response_dynamics(Evaluator(p), np.array([0, 1, 2, 3]))
    assert res.converged
    for i in range(4):
        assert p.utility_replace(res.placement, i).max() <= res.value + 1e-12
    # budget too small for a single sweep: nothing is evaluated
    ev = Evaluator(p, budget=5)
    res = team.with_restarts(
        lambda e, x, r: team.sampled_best_response(e, x, r, n_random=2, n_near=1),
        ev,
        4,
        np.random.default_rng(0),
    )
    assert ev.n_evals <= 5


def test_sampled_best_response_never_decreases_the_utility():
    p = random_problem(3, n_sites=30)
    ev = Evaluator(p, budget=600)
    init = np.array([0, 1, 2, 3, 4])
    res = team.sampled_best_response(ev, init, np.random.default_rng(1), n_random=4, n_near=3)
    assert res.value >= p.utility(init[None, :])[0]
    assert res.value == pytest.approx(p.utility(res.placement[None, :])[0])
    assert res.extra["evals_at_last_move"] <= res.n_evals


@pytest.mark.parametrize("reward", ["team", "difference"])
def test_bandit_learners_improve_over_random_placements(reward):
    p = random_problem(4, n_sites=12)
    rng = np.random.default_rng(0)
    random_mean = p.utility(rng.integers(12, size=(500, 3))).mean()
    res = team.bandit_learning(
        Evaluator(p, budget=6000),
        3,
        np.random.default_rng(5),
        reward=reward,
        batch=16,
        learning_rate=1.0,
    )
    assert res.value > random_mean
    assert res.extra["samples"] * (4 if reward == "difference" else 1) + 1 == res.n_evals


def test_simulated_annealing_at_zero_temperature_is_a_better_response_walk():
    p = random_problem(6, n_sites=15)
    ev = Evaluator(p, budget=300)
    init = np.array([0, 1, 2])
    res = cen.simulated_annealing(ev, init, np.random.default_rng(2), t_start=1e-30, t_end=1e-30)
    values = [v for _, v in ev.trace]
    assert res.value == pytest.approx(ev.best_value)  # never accepts a worse placement
    assert values == sorted(values) and res.n_evals == 300


def test_determinism_of_every_method():
    """Same seed, same result: placements, values and evaluation counts are identical."""
    p = random_problem(8, n_sites=25)
    settings = ex.PASSES[ex.FINAL_PASS]
    init = np.array([1, 2, 3, 4, 5])
    for name in ex.METHODS:
        runs = [
            ex.run_method(name, p, 5, 700, np.random.default_rng([4, 2]), settings, init)
            for _ in range(2)
        ]
        assert np.array_equal(runs[0].best_placement, runs[1].best_placement), name
        assert runs[0].best_value == runs[1].best_value and runs[0].trace == runs[1].trace, name
        other = ex.run_method(name, p, 5, 700, np.random.default_rng([4, 3]), settings, init)
        assert other.n_evals <= 700


def test_determinism_of_an_experiment_task(scene):
    """A full experiment task gives identical records when repeated (also across processes:
    every task derives its random streams from its own arguments)."""
    settings = ex.quick(ex.PASSES[ex.FINAL_PASS])
    args = (settings, 0, 3)
    first = ex._e2_task(args)
    second = ex._e2_task(args)
    assert first == second
    serial = ex.parallel_map(ex._e2_task, [args, (settings, 1, 3)], jobs=1)
    assert serial[0] == first
