"""Numerical checks of Propositions 1-5 of ``docs/theory.md`` on small random instances."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from conftest import random_problem

from cfisac_ma import centralized as cen
from cfisac_ma import team
from cfisac_ma.objective import Evaluator, Problem

TOL = 1e-10


def all_profiles(n_sites: int, n_agents: int) -> np.ndarray:
    return np.array(list(itertools.product(range(n_sites), repeat=n_agents)))


def subsets_of(n: int):
    for size in range(n + 1):
        yield from itertools.combinations(range(n), size)


def set_values(f, n: int) -> dict[frozenset, float]:
    """Value of every subset of ``range(n)``."""
    subs = list(subsets_of(n))
    padded = np.full((len(subs), n), -1)
    for i, s in enumerate(subs):
        padded[i, : len(s)] = s
    return dict(zip(map(frozenset, subs), f(padded), strict=True))


def is_submodular(values: dict[frozenset, float], n: int, tol: float = TOL) -> bool:
    """Exhaustive check of f(S + a) - f(S) >= f(S + b + a) - f(S + b) for all S, a != b."""
    for s in values:
        rest = [x for x in range(n) if x not in s]
        for a, b in itertools.permutations(rest, 2):
            if values[s | {a}] - values[s] < values[s | {a, b}] - values[s | {b}] - tol:
                return False
    return True


def is_monotone(values: dict[frozenset, float], n: int, tol: float = TOL) -> bool:
    return all(values[s | {a}] >= values[s] - tol for s in values for a in range(n) if a not in s)


def nash_mask(problem: Problem, profiles: np.ndarray, payoff: str = "team") -> np.ndarray:
    """Pure Nash equilibria among all profiles, by checking every unilateral deviation."""
    n_agents = profiles.shape[1]
    is_ne = np.ones(len(profiles), dtype=bool)
    for j, a in enumerate(profiles):
        for i in range(n_agents):
            values = problem.utility_replace(a, i)
            if payoff == "difference":
                removed = a.copy()
                removed[i] = -1
                values = values - problem.utility(removed[None, :])[0]
            if values.max() > values[a[i]] + TOL:
                is_ne[j] = False
                break
    return is_ne


# ----------------------------------------------------------------------------- Proposition 1
@pytest.mark.parametrize("seed", range(4))
def test_p1_team_game_is_an_exact_potential_game(seed):
    """Unilateral deviations change every agent's utility and the potential by the same
    amount, and best-response dynamics end in a pure Nash equilibrium."""
    p = random_problem(seed, n_sites=7)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        a = rng.integers(7, size=3)
        i, c = int(rng.integers(3)), int(rng.integers(7))
        b = a.copy()
        b[i] = c
        # team utility of the deviating agent is U itself: the potential difference is exact
        du = p.utility(b[None, :])[0] - p.utility(a[None, :])[0]
        assert p.utility_replace(a, i)[c] - p.utility_replace(a, i)[a[i]] == pytest.approx(
            du, abs=1e-12
        )
    profiles = all_profiles(7, 3)
    is_ne = nash_mask(p, profiles)
    index = {tuple(pr): j for j, pr in enumerate(profiles)}
    for start in rng.integers(7, size=(10, 3)):
        ev = Evaluator(p)
        res = team.best_response_dynamics(ev, start)
        assert res.converged and is_ne[index[tuple(res.placement)]]
        assert res.n_moves <= 7**3  # no profile is visited twice
        improvements = [v for _, v in ev.trace]
        assert all(b > a for a, b in zip(improvements, improvements[1:], strict=False))
    # the maximiser of the potential is an equilibrium
    values = p.utility(profiles)
    assert is_ne[int(np.argmax(values))]


# ----------------------------------------------------------------------------- Proposition 2
@pytest.mark.parametrize("seed", range(4))
def test_p2_difference_utilities_give_the_same_game(seed):
    p = random_problem(seed, n_sites=6)
    profiles = all_profiles(6, 3)
    assert np.array_equal(nash_mask(p, profiles, "team"), nash_mask(p, profiles, "difference"))
    rng = np.random.default_rng(seed)
    for start in rng.integers(6, size=(10, 3)):
        r_team = team.best_response_dynamics(Evaluator(p), start, reward="team")
        r_diff = team.best_response_dynamics(Evaluator(p), start, reward="difference")
        assert np.array_equal(r_team.placement, r_diff.placement)
        assert r_team.n_moves == r_diff.n_moves
        # one extra evaluation (the counterfactual) per sweep
        assert r_diff.n_evals == r_team.n_evals + r_team.n_evals // p.n_sites
    # the difference utility is the effect of removing the agent
    a = np.array([0, 3, 5])
    for i in range(3):
        removed = a.copy()
        removed[i] = -1
        d = p.utility(a[None, :])[0] - p.utility(removed[None, :])[0]
        assert d >= -TOL


# ----------------------------------------------------------------------------- Proposition 3
@pytest.mark.parametrize("seed", range(5))
def test_p3_sum_rate_is_monotone_submodular(seed):
    p = random_problem(seed, n_sites=7, lam=1.0)
    values = set_values(p.utility, 7)
    assert values[frozenset()] == pytest.approx(0.0, abs=1e-12)
    assert is_monotone(values, 7) and is_submodular(values, 7)


def test_p3_min_rate_is_not_submodular():
    # two users, two sites, each site serves one user: min rate is 0, 0, 0, positive
    snr = np.array([[100.0, 0.0], [0.0, 100.0]])
    w = np.ones((2, 1))
    u = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    p = Problem(snr, w, u, lam=1.0, rate="maxmin")
    values = set_values(p.utility, 2)
    assert values[frozenset({0})] == 0.0 and values[frozenset({1})] == 0.0
    assert values[frozenset({0, 1})] == pytest.approx(1.0)
    assert not is_submodular(values, 2)


# ----------------------------------------------------------------------------- Proposition 4
@pytest.mark.parametrize("seed", range(5))
def test_p4_split_sensing_is_submodular_in_the_transmit_set(seed):
    p = random_problem(seed, n_sites=9, lam=0.0)
    rx = np.array([7, 8])

    def f(tx_sets):
        split = p.with_options(sensing="split", n_tx=tx_sets.shape[1])
        rx_block = np.repeat(rx[None, :], len(tx_sets), axis=0)
        return split.utility(np.concatenate([tx_sets, rx_block], axis=1))

    values = set_values(f, 7)  # transmit sets among sites 0..6, receive set {7, 8}
    assert values[frozenset()] == pytest.approx(0.0, abs=1e-12)
    assert is_monotone(values, 7) and is_submodular(values, 7)

    def g(rx_sets):  # symmetric statement: fixed transmit set, varying receive set
        split = p.with_options(sensing="split", n_tx=2)
        tx_block = np.repeat(rx[None, :], len(rx_sets), axis=0)
        return split.utility(np.concatenate([tx_block, rx_sets], axis=1))

    values = set_values(g, 7)
    assert is_monotone(values, 7) and is_submodular(values, 7)


def test_p4_joint_sensing_counterexample():
    """Two APs in orthogonal directions, unit gains, eps = 1 (the example of the theory
    document): the gain of the second AP is log 9 after the first and log 5 alone."""
    w = np.ones((2, 1))
    u = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    p = Problem(np.zeros((2, 1)), w, u, lam=0.0, eps=1.0)

    def logdet(sets):
        return p.components(np.array(sets)).logdet[0]

    assert logdet([[-1, -1]]) == pytest.approx(0.0, abs=1e-12)
    assert logdet([[0, -1]]) == pytest.approx(math.log(5.0))
    assert logdet([[1, -1]]) == pytest.approx(math.log(5.0))
    assert logdet([[0, 1]]) == pytest.approx(math.log(45.0))
    gain_alone = logdet([[1, -1]]) - logdet([[-1, -1]])
    gain_after = logdet([[0, 1]]) - logdet([[0, -1]])
    assert gain_after == pytest.approx(math.log(9.0)) and gain_after > gain_alone


def test_p4_joint_sensing_violations_on_random_instances():
    violations = 0
    for seed in range(5):
        p = random_problem(seed, n_sites=7, lam=0.0)
        values = set_values(p.utility, 7)
        assert is_monotone(values, 7)
        violations += not is_submodular(values, 7)
    assert violations > 0


# ----------------------------------------------------------------------------- Proposition 5
def exact_ratios(values: dict[frozenset, float], n: int) -> tuple[float, float]:
    """Exact submodularity ratio and diminishing-return ratio of a monotone set function."""
    gamma, gamma_d = np.inf, np.inf
    for s in values:
        rest = [x for x in range(n) if x not in s]
        for size in range(1, len(rest) + 1):
            for omega in itertools.combinations(rest, size):
                joint = values[s | set(omega)] - values[s]
                if joint > TOL:
                    gamma = min(gamma, sum(values[s | {w}] - values[s] for w in omega) / joint)
        for a in rest:
            for size in range(1, len(rest)):
                for extra in itertools.combinations([x for x in rest if x != a], size):
                    t = s | set(extra)
                    big = values[t | {a}] - values[t]
                    if big > TOL:
                        gamma_d = min(gamma_d, (values[s | {a}] - values[s]) / big)
    return min(gamma, 1.0), min(gamma_d, 1.0)


@pytest.mark.parametrize("seed", range(6))
def test_p5_equilibria_and_greedy_for_submodular_utility(seed):
    """Sum-rate utility (monotone submodular): every equilibrium is at least half the
    optimum and greedy is at least (1 - 1/e) of it."""
    p = random_problem(seed, n_sites=9, lam=1.0)
    for n_agents in (2, 3):
        subsets, values = cen.exhaustive_values(p, n_agents)
        opt = values.max()
        is_ne = cen.nash_equilibria(subsets, values, p.n_sites)
        assert is_ne.any() and is_ne[int(np.argmax(values))]
        assert values[is_ne].min() >= 0.5 * opt - TOL
        g = cen.greedy(Evaluator(p), n_agents)
        assert g.value >= (1 - 1 / math.e) * opt - TOL
        # equilibria including profiles with repeated sites
        profiles = all_profiles(p.n_sites, n_agents)
        ne_all = nash_mask(p, profiles)
        assert p.utility(profiles)[ne_all].min() >= 0.5 * opt - TOL


@pytest.mark.parametrize("seed", range(4))
def test_p5_general_bound_with_exact_ratios(seed):
    """Joint utility (not submodular): U(NE) >= g / (1 + g) * U(OPT), g = gamma * gamma_d^2."""
    p = random_problem(seed, n_sites=6, lam=0.3)
    values = set_values(p.utility, 6)
    gamma, gamma_d = exact_ratios(values, 6)
    g = gamma * gamma_d**2
    for n_agents in (2, 3):
        profiles = all_profiles(6, n_agents)
        u = p.utility(profiles)
        ne = nash_mask(p, profiles)
        assert u[ne].min() >= g / (1 + g) * u.max() - TOL


def test_nash_enumeration_matches_brute_force():
    p = random_problem(11, n_sites=8)
    subsets, values = cen.exhaustive_values(p, 3)
    fast = cen.nash_equilibria(subsets, values, 8)
    profiles = all_profiles(8, 3)
    slow = nash_mask(p, profiles)
    distinct = np.array([len(set(pr)) == 3 for pr in profiles])
    slow_sets = {frozenset(pr) for pr in profiles[slow & distinct]}
    fast_sets = {frozenset(s) for s in subsets[fast]}
    assert slow_sets == fast_sets
