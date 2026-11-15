"""Numerical checks of Propositions 6-9 of ``docs/theory.md`` (mode-selection game) on small
random instances, exhaustively where the instance size allows."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from conftest import random_mode_problem

from cfisac_ma import operation as op
from cfisac_ma.aerial import fim_matrix
from cfisac_ma.objective import _split_entries

TOL = 1e-10


def index_of(profile) -> int:
    return int(np.ravel_multi_index(tuple(int(a) for a in profile), (3,) * len(profile)))


def set_profile(n_agents: int, tx=(), rx=()) -> np.ndarray:
    profile = np.zeros(n_agents, dtype=np.int64)
    profile[list(tx)] = op.TX
    profile[list(rx)] = op.RX
    return profile


def subsets(items):
    items = list(items)
    for size in range(len(items) + 1):
        yield from itertools.combinations(items, size)


def is_monotone_submodular(f, ground) -> bool:
    """Exhaustive check of f(S + a) >= f(S) and of diminishing returns on ``ground``."""
    values = {frozenset(s): f(s) for s in subsets(ground)}
    for s in values:
        rest = [x for x in ground if x not in s]
        for a in rest:
            if values[s | {a}] < values[s] - TOL:
                return False
            for b in rest:
                if (
                    b != a
                    and values[s | {a}] - values[s] < values[s | {a, b}] - values[s | {b}] - TOL
                ):
                    return False
    return True


# ----------------------------------------------------------------------------- Proposition 6
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("costs", [(0.0, 0.0), (0.01, 0.005), (0.3, 0.2)])
def test_p6_mode_game_is_an_exact_potential_game(seed, costs):
    """Every agent's utility is U, so U is an exact potential for any costs; improvement
    paths are finite and end in a Nash equilibrium; the maximiser is an equilibrium."""
    p = random_mode_problem(seed, 6, c_tx=costs[0], c_rx=costs[1])
    profiles = op.all_profiles(6)
    values = p.utility(profiles)
    nash, stable = op.equilibrium_masks(values, 6)
    assert nash[int(np.argmax(values))] and stable[int(np.argmax(values))]
    rng = np.random.default_rng(seed)
    for _ in range(8):
        start = rng.integers(3, size=6)
        # the change of the common utility under a unilateral deviation is the change of U
        i, mode = int(rng.integers(6)), int(rng.integers(3))
        other = start.copy()
        other[i] = mode
        dev = p.deviations(start)
        assert dev[i, mode] - dev[i, start[i]] == pytest.approx(
            values[index_of(other)] - values[index_of(start)], abs=1e-12
        )
        for method in (op.best_response, op.coloured_best_response, op.pair_best_response):
            res = method(p, start)
            assert res.converged and nash[index_of(res.profile)]
            assert res.n_moves < 3**6  # no profile is visited twice
            steps = [v for _, v in res.trace]
            assert all(b > a for a, b in zip(steps, steps[1:], strict=False))
            assert res.value == pytest.approx(values[index_of(res.profile)], abs=1e-12)
        assert stable[index_of(op.pair_best_response(p, start).profile)]
    g = op.greedy(p)
    assert g.converged and nash[index_of(g.profile)]


# ----------------------------------------------------------------------------- Proposition 7
@pytest.mark.parametrize("seed", range(4))
def test_p7_difference_utility_needs_only_the_agents_neighbourhood(seed):
    """The gain of a mode change is computed from the SNR sums of the users the agent
    reaches and the role aggregates of the targets it sees; modes of agents that share
    neither a user nor a target with it do not enter."""
    p = random_mode_problem(seed, 12, 8, 5, reach=0.2, sees=0.25)
    adj = p.interaction_graph()
    assert 0 < adj.sum() < 12 * 11  # neither empty nor complete
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(6):
        profile = rng.integers(3, size=12)
        base = p.utility(profile[None, :])[0]
        for i in range(12):
            strangers = np.flatnonzero(~adj[i] & (np.arange(12) != i))
            shuffled = profile.copy()
            shuffled[strangers] = rng.integers(3, size=len(strangers))
            for mode in range(3):
                changed = profile.copy()
                changed[i] = mode
                gain = p.utility(changed[None, :])[0] - base
                assert p.local_gain(profile, i, mode) == pytest.approx(gain, abs=1e-12)
                # same gain whatever the non-neighbours do
                again = shuffled.copy()
                again[i] = mode
                far = p.utility(again[None, :])[0] - p.utility(shuffled[None, :])[0]
                assert far == pytest.approx(gain, abs=1e-12)
                assert p.local_gain(shuffled, i, mode) == pytest.approx(gain, abs=1e-12)
                checked += len(strangers) > 0
    assert checked > 100
    # per role and target the aggregate is (W, b, A): ten numbers
    assert op.N_FEATURES == 10 and p.feat.shape[-1] == 10
    assert np.array_equal(p.step_work(), 2 * p.reach.sum(axis=1) + 3 * p.sees.sum(axis=1))


def test_p7_role_aggregates_give_the_information_matrix():
    p = random_mode_problem(1, 7)
    profile = np.array([1, 2, 1, 0, 2, 1, 2])
    _, ft, fr, _, _ = p.aggregates(profile[None, :])
    fim = fim_matrix(_split_entries(ft[0], fr[0]))
    for q in range(p.n_targets):
        ref = np.zeros((3, 3))
        for m in np.flatnonzero(profile == op.TX):
            for n in np.flatnonzero(profile == op.RX):
                s = p.u[m, q] + p.u[n, q]
                ref += p.w[m, q] * p.w[n, q] * np.outer(s, s)
        assert np.allclose(fim[q], ref, rtol=1e-10, atol=1e-12)


# ----------------------------------------------------------------------------- Proposition 8
@pytest.mark.parametrize("seed", range(5))
def test_p8_simultaneous_moves_of_non_adjacent_agents_add_up(seed):
    p = random_mode_problem(seed, 14, 9, 5, reach=0.15, sees=0.2)
    colours = op.greedy_colouring(p.interaction_graph())
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(colours == c) for c in range(colours.max() + 1)]
    assert max(len(c) for c in classes) >= 2
    for _ in range(10):
        profile = rng.integers(3, size=14)
        base = p.utility(profile[None, :])[0]
        for members in classes:
            joint = profile.copy()
            joint[members] = rng.integers(3, size=len(members))
            single = 0.0
            for i in members:
                one = profile.copy()
                one[i] = joint[i]
                single += p.utility(one[None, :])[0] - base
            assert p.utility(joint[None, :])[0] - base == pytest.approx(single, abs=1e-12)
    # best response on the colouring is an improvement process that ends in an equilibrium
    for _ in range(5):
        start = rng.integers(3, size=14)
        res = op.coloured_best_response(p, start, colours=colours)
        assert res.converged and op.deviation_gain(p, res.profile) <= op.TOL
        steps = [v for _, v in res.trace]
        assert all(b > a for a, b in zip(steps, steps[1:], strict=False))
        assert res.n_subrounds >= colours.max() + 1  # the last cycle certifies the equilibrium
        assert res.n_evals == 3 * res.n_steps


def test_p8_additivity_fails_for_adjacent_agents():
    """Two idle APs that see the same target: each alone gains nothing, together they do."""
    p = random_mode_problem(0, 4, 2, 1, reach=0.0, sees=1.0, lam=0.0, c_tx=0.0, c_rx=0.0)
    assert p.interaction_graph()[0, 1]
    off = op.all_off(p)
    assert p.utility(set_profile(4, tx=[0])[None, :])[0] == 0.0
    assert p.utility(set_profile(4, rx=[1])[None, :])[0] == 0.0
    assert p.utility(set_profile(4, tx=[0], rx=[1])[None, :])[0] > 0.1
    assert p.utility(off[None, :])[0] == 0.0


# ----------------------------------------------------------------------------- Proposition 9
@pytest.mark.parametrize("seed", range(4))
def test_p9_rate_and_single_role_sensing_are_monotone_submodular(seed):
    p = random_mode_problem(seed, 7, 6, 3, sees=0.8)
    pool, fixed = [0, 1, 2, 3, 4], [5, 6]

    def rate(tx):
        return float(p.terms(set_profile(7, tx=tx)[None, :])["rate"][0])

    def sens_tx(tx):  # transmit set varies, receive set fixed
        return float(p.terms(set_profile(7, tx=tx, rx=fixed)[None, :])["sens"][0].sum())

    def sens_rx(rx):  # receive set varies, transmit set fixed
        return float(p.terms(set_profile(7, tx=fixed, rx=rx)[None, :])["sens"][0].sum())

    assert rate(()) == 0.0 and is_monotone_submodular(rate, range(7))
    assert is_monotone_submodular(sens_tx, pool) and is_monotone_submodular(sens_rx, pool)
    # the line-of-sight mask only removes terms: a target nobody sees has value 0
    blind = random_mode_problem(seed, 7, 6, 3, sees=0.0)
    assert np.all(blind.terms(np.array([[1, 2, 1, 2, 1, 2, 1]]))["sens"] == 0.0)


def test_p9_matrix_increments_are_complementary_across_roles_but_the_value_is_not():
    """J(T + m, R + n) - J(T, R + n) - J(T + m, R) + J(T, R) is the pair matrix of (m, n),
    which is positive semidefinite; the same second difference of s_q takes both signs."""
    positive = negative = 0
    for seed in range(6):
        p = random_mode_problem(seed, 8, 2, 2, sees=1.0)
        rng = np.random.default_rng(seed)

        def info(tx, rx, p=p):
            _, ft, fr, _, _ = p.aggregates(set_profile(8, tx=tx, rx=rx)[None, :])
            return fim_matrix(_split_entries(ft[0], fr[0])), op.sensing_values(ft[0], fr[0], p.eps)

        for _ in range(40):
            perm = rng.permutation(8)
            m, n = int(perm[0]), int(perm[1])
            n_tx, n_rx = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            tx, rx = list(perm[2 : 2 + n_tx]), list(perm[5 : 5 + min(n_rx, 3)])
            (j00, s00), (j10, s10) = info(tx, rx), info([*tx, m], rx)
            (j01, s01), (j11, s11) = info(tx, [*rx, n]), info([*tx, m], [*rx, n])
            cross = j11 - j10 - j01 + j00
            for q in range(2):
                s = p.u[m, q] + p.u[n, q]
                assert np.allclose(cross[q], p.w[m, q] * p.w[n, q] * np.outer(s, s), atol=1e-9)
                assert np.linalg.eigvalsh(cross[q]).min() >= -1e-9
            second = (s11 - s10 - s01 + s00).sum()
            positive += second > 1e-9
            negative += second < -1e-9
    assert positive > 0 and negative > 0


@pytest.mark.parametrize("costs", [(0.0, 0.0), (0.004, 0.002)])
def test_p9_all_off_is_an_equilibrium_with_ratio_zero(costs):
    """Sensing only: a lone transmitter or receiver gains nothing, so all-off is a Nash
    equilibrium of utility 0 although the optimum is positive; pair moves leave it."""
    p = random_mode_problem(3, 6, lam=0.0, sees=0.9, c_tx=costs[0], c_rx=costs[1])
    off = op.all_off(p)
    values = p.utility(op.all_profiles(6))
    nash, stable = op.equilibrium_masks(values, 6)
    assert values[index_of(off)] == 0.0 and nash[index_of(off)] and not stable[index_of(off)]
    assert values.max() > 0.3
    assert op.deviation_gain(p, off) <= 0.0
    for stuck in (op.best_response(p, off), op.greedy(p), op.coloured_best_response(p, off)):
        assert stuck.converged and stuck.value == 0.0 and stuck.n_moves == 0
    pair = op.pair_best_response(p, off)
    assert pair.converged and pair.value > 0.9 * values.max() and pair.extra["pair_moves"] >= 1
    assert stable[index_of(pair.profile)]
    # the optimum can be scaled at will while the equilibrium stays at 0: no factor exists
    better = p.with_options(s_ref=p.s_ref / 10.0)
    assert better.utility(off[None, :])[0] == 0.0 and op.deviation_gain(better, off) <= 0.0
    assert better.utility(op.all_profiles(6)).max() > 3.0


@pytest.mark.parametrize("seed", range(6))
def test_p9_pair_stable_profiles_leave_no_profitable_idle_pair(seed):
    """In a pair-stable profile, a target without a Tx-Rx pair in line of sight has no two
    idle APs that see it and whose lone-pair value exceeds c_tx + c_rx."""
    costs = (0.02, 0.01) if seed % 2 else (0.15, 0.1)
    p = random_mode_problem(seed, 6, 4, 3, sees=0.6, c_tx=costs[0], c_rx=costs[1])
    profiles = op.all_profiles(6)
    values = p.utility(profiles)
    nash, stable = op.equilibrium_masks(values, 6)
    paired = p.terms(profiles)["paired"]

    def worst_idle_pair(profile, q):
        idle = [i for i in range(6) if profile[i] == op.OFF and p.sees[i, q]]
        gains = [op.pair_gain_bound(p, m, n, q) for m, n in itertools.combinations(idle, 2)]
        return max(gains, default=-np.inf)

    cases = 0
    for j in np.flatnonzero(stable):
        for q in np.flatnonzero(~paired[j]):
            worst = worst_idle_pair(profiles[j], q)
            assert worst <= p.c_tx + p.c_rx + 1e-9
            cases += worst > -np.inf
    if seed % 2 == 0:  # with the high costs the premise occurs (the check is not vacuous)
        assert cases > 0
    # the bound is the value of the lone pair: k_sens * log(1 + w_m w_n |u_m + u_n|^2 / eps)
    m, n, q = (
        0,
        1,
        int(np.flatnonzero(p.sees[0] & p.sees[1])[0]) if (p.sees[0] & p.sees[1]).any() else 0,
    )
    s = p.u[m, q] + p.u[n, q]
    closed = p.k_sens * math.log(1.0 + p.w[m, q] * p.w[n, q] * (s @ s) / p.eps)
    assert op.pair_gain_bound(p, m, n, q) == pytest.approx(closed, rel=1e-9, abs=1e-12)
    assert op.pair_gain_bound(p, n, m, q) == pytest.approx(closed, rel=1e-9, abs=1e-12)


def test_p9_unilateral_equilibria_can_violate_the_pair_bound():
    """The statement of the previous test is false for Nash equilibria that are not
    pair-stable: all-off at lam = 0 has profitable idle pairs."""
    p = random_mode_problem(3, 6, lam=0.0, sees=0.9, c_tx=0.004, c_rx=0.002)
    best = max(
        op.pair_gain_bound(p, m, n, q)
        for q in range(p.n_targets)
        for m, n in itertools.combinations(range(6), 2)
        if p.sees[m, q] and p.sees[n, q]
    )
    assert best > p.c_tx + p.c_rx
    assert op.deviation_gain(p, op.all_off(p)) <= 0.0
