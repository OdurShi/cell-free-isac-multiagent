"""The mode-selection game: evaluation, methods, episodes."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from conftest import random_mode_problem

from cfisac_ma import aerial
from cfisac_ma import experiments_op as exo
from cfisac_ma import operation as op


def loop_utility(p: op.ModeProblem, profile) -> float:
    """Direct transcription of the utility: double sum over transmitters and receivers."""
    tx = [i for i, a in enumerate(profile) if a == op.TX]
    rx = [i for i, a in enumerate(profile) if a == op.RX]
    rate = sum(math.log2(1.0 + sum(p.snr[i, k] for i in tx)) for k in range(p.n_users))
    sens = 0.0
    for q in range(p.n_targets):
        fim = np.zeros((3, 3))
        for m in tx:
            for n in rx:
                s = p.u[m, q] + p.u[n, q]
                fim += p.w[m, q] * p.w[n, q] * np.outer(s, s)
        sens += np.linalg.slogdet(fim + p.eps * np.eye(3))[1] - 3.0 * math.log(p.eps)
    return (
        p.lam * rate / p.r_ref
        + (1.0 - p.lam) * sens / (p.n_targets * p.s_ref)
        - p.c_tx * len(tx)
        - p.c_rx * len(rx)
    )


@pytest.mark.parametrize("seed", range(4))
def test_utility_matches_the_double_sum(seed):
    p = random_mode_problem(seed, 7)
    rng = np.random.default_rng(seed)
    profiles = rng.integers(3, size=(30, 7))
    values = p.utility(profiles)
    for profile, value in zip(profiles, values, strict=True):
        assert value == pytest.approx(loop_utility(p, profile), rel=1e-9, abs=1e-12)
    assert p.utility(op.all_off(p)[None, :])[0] == 0.0
    terms = p.terms(profiles)
    assert np.allclose(terms["utility"], values, rtol=0, atol=1e-15)
    assert np.allclose(terms["n_tx"], (profiles == op.TX).sum(axis=1))
    # normalisers: every AP transmitting; joint-role value of all APs at the targets
    assert p.r_ref == pytest.approx(np.log2(1 + p.snr.sum(axis=0)).sum())
    total = p.feat.sum(axis=0)
    assert p.s_ref == pytest.approx(op.sensing_values(total, total, p.eps).mean())


@pytest.mark.parametrize("seed", range(4))
def test_incremental_evaluation_matches_direct_evaluation(seed):
    p = random_mode_problem(seed, 8, 6, 3)
    rng = np.random.default_rng(seed)
    edges = np.array(list(itertools.combinations(range(8), 2)))
    for _ in range(10):
        profile = rng.integers(3, size=8)
        dev = p.deviations(profile)
        pair = p.pair_deviations(profile, edges, chunk=5)
        base = p.utility(profile[None, :])[0]
        for i in range(8):
            for mode in range(3):
                other = profile.copy()
                other[i] = mode
                direct = p.utility(other[None, :])[0]
                assert dev[i, mode] == pytest.approx(direct, rel=1e-12, abs=1e-13)
                assert p.local_gain(profile, i, mode) == pytest.approx(direct - base, abs=1e-12)
        for (i, j), block in zip(edges, pair, strict=True):
            for mi in range(3):
                for mj in range(3):
                    other = profile.copy()
                    other[i], other[j] = mi, mj
                    assert block[mi, mj] == pytest.approx(
                        p.utility(other[None, :])[0], rel=1e-12, abs=1e-13
                    )


def test_exhaustive_terms_reproduce_the_utility():
    p = random_mode_problem(5, 6)
    profiles = op.all_profiles(6)
    assert profiles.shape == (729, 6)
    assert [tuple(r) for r in profiles] == list(itertools.product(range(3), repeat=6))
    terms = op.exhaustive_terms(p, chunk=100)
    assert np.allclose(op.utility_from_terms(p, terms), p.utility(profiles), atol=1e-13)
    for lam, c_tx, c_rx in ((0.0, 0.0, 0.0), (1.0, 0.02, 0.0), (0.3, 0.004, 0.002)):
        q = p.with_options(lam=lam, c_tx=c_tx, c_rx=c_rx)
        assert np.allclose(op.utility_from_terms(q, terms), q.utility(profiles), atol=1e-13)
    paired = p.terms(profiles)["paired"].sum(axis=1)
    assert np.array_equal(terms["paired"], paired)


def test_equilibrium_masks_match_brute_force():
    for seed in range(3):
        p = random_mode_problem(seed, 5)
        profiles = op.all_profiles(5)
        values = p.utility(profiles)
        nash, stable = op.equilibrium_masks(values, 5)
        index = {tuple(r): j for j, r in enumerate(profiles)}
        for j, profile in enumerate(profiles):
            single = max(
                values[index[tuple(np.where(np.arange(5) == i, m, profile))]]
                for i in range(5)
                for m in range(3)
            )
            assert nash[j] == (values[j] >= single - op.TOL)
            double = values[j]
            for i, k in itertools.combinations(range(5), 2):
                for mi, mk in itertools.product(range(3), repeat=2):
                    other = profile.copy()
                    other[i], other[k] = mi, mk
                    double = max(double, values[index[tuple(other)]])
            assert stable[j] == (values[j] >= double - op.TOL)
        assert nash[int(np.argmax(values))] and stable[int(np.argmax(values))]
        assert not (stable & ~nash).any()


def test_clustering_and_failures():
    p = random_mode_problem(2, 12, 6, 4, reach=0.9, sees=0.9)
    c = p.clustered(3)
    assert c.reach.sum(axis=0).max() <= 3 and c.sees.sum(axis=0).max() <= 3
    assert (c.r_ref, c.s_ref) == (p.r_ref, p.s_ref)
    for k in range(6):  # a user keeps its strongest APs
        kept = np.flatnonzero(c.snr[:, k])
        assert p.snr[kept, k].min() >= np.delete(p.snr[:, k], kept).max()
    assert c.interaction_graph().sum() < p.interaction_graph().sum()
    failed = np.array([1, 7])
    f = p.without(failed)
    assert not f.free[failed].any() and f.free.sum() == 10
    profile = np.random.default_rng(0).integers(3, size=12)
    alive = profile.copy()
    alive[failed] = op.OFF
    # a failed AP contributes nothing in any mode (it would only pay its cost)
    assert f.utility(alive[None, :])[0] == pytest.approx(p.utility(alive[None, :])[0], abs=1e-14)
    assert f.degrees()[failed].sum() == 0
    res = op.best_response(f, alive)
    assert (res.profile[failed] == op.OFF).all()
    assert (op.random_profile(f, np.random.default_rng(1))[failed] == op.OFF).all()


def test_colouring_is_proper_and_bounded_by_the_clique():
    for seed in range(5):
        p = random_mode_problem(seed, 14, 6, 3, reach=0.15, sees=0.2)
        adj = p.interaction_graph()
        assert np.array_equal(adj, adj.T) and not adj.diagonal().any()
        colours = op.greedy_colouring(adj)
        i, j = np.nonzero(adj)
        assert (colours[i] != colours[j]).all()
        assert colours.max() + 1 >= op.clique_lower_bound(p)
        assert colours.max() + 1 <= adj.sum(axis=1).max() + 1
        r, s = p.reach.astype(int), p.sees.astype(int)
        ref = ((r @ r.T) + (s @ s.T)) > 0
        np.fill_diagonal(ref, False)
        assert np.array_equal(adj, ref)


# ------------------------------------------------------------------------------ methods
def test_evaluation_counts_and_budgets():
    p = random_mode_problem(3, 9)
    rng = np.random.default_rng(0)
    init = op.random_profile(p, rng)
    res = op.best_response(p, init)
    assert res.converged and res.n_evals == 3 * res.n_steps
    assert res.work <= res.n_steps * (2 * p.n_users + 3 * p.n_targets)
    capped = op.best_response(p, init, budget=10)
    assert capped.n_evals == 9 and not capped.converged
    g = op.greedy(p)
    assert g.converged and g.n_evals == 2 * 9 * g.n_steps and g.n_moves == g.n_steps - 1
    assert op.deviation_gain(p, g.profile) <= op.TOL  # greedy stops at an equilibrium
    sa = op.simulated_annealing(p, init, 200, rng, t_start=0.0, t_end=0.0)
    assert sa.n_evals == 200
    values = [v for _, v in sa.trace]
    assert values == sorted(values) and sa.value == pytest.approx(sa.best_value)
    rs = op.random_search(p, 300, rng, batch=64)
    assert rs.n_evals == 300 and rs.trace[-1][0] <= 300
    assert rs.best_value == pytest.approx(p.utility(rs.best_profile[None, :])[0])
    pair = op.pair_best_response(p, init, budget=50)
    assert pair.n_evals <= 50
    multi = op.with_restarts(op.best_response, p, init, 400, rng)
    assert multi.n_evals <= 400 and multi.best_value >= multi.extra["first"].value
    lll = op.log_linear_learning(p, op.all_off(p), 300, rng)
    assert lll.n_evals <= 300 or lll.converged
    for r in (res, g, sa, rs, pair, multi, lll):
        assert p.utility(r.best_profile[None, :])[0] == pytest.approx(r.best_value, abs=1e-12)
        assert r.trace[-1][1] == pytest.approx(r.best_value)


def test_methods_are_deterministic():
    p = random_mode_problem(4, 10)
    init = op.random_profile(p, np.random.default_rng(9))

    def run_all(seed):
        rng = np.random.default_rng(seed)
        return [
            op.best_response(p, init),
            op.coloured_best_response(p, init),
            op.pair_best_response(p, init),
            op.greedy(p),
            op.simulated_annealing(p, init, 300, rng, t_start=1e-2, t_end=1e-3),
            op.log_linear_learning(p, init, 300, rng),
            op.random_search(p, 300, rng),
            op.reference_solve(p, rng),
        ]

    for a, b in zip(run_all(1), run_all(1), strict=True):
        assert np.array_equal(a.best_profile, b.best_profile) and a.best_value == b.best_value
        assert a.n_evals == b.n_evals and a.trace == b.trace and a.messages == b.messages


def test_reference_solver_is_at_least_as_good_as_its_parts():
    for seed in range(4):
        p = random_mode_problem(seed, 9)
        ref = op.reference_solve(p, np.random.default_rng(seed))
        single = op.pair_best_response(p, op.all_off(p))
        assert ref.value >= single.value - op.TOL and ref.n_evals >= single.n_evals
        values = p.utility(op.all_profiles(9))
        assert ref.value <= values.max() + 1e-12
        _, stable = op.equilibrium_masks(values, 9)
        index = int(np.ravel_multi_index(tuple(ref.profile), (3,) * 9))
        assert stable[index]


# ------------------------------------------------------------------------------ episodes
@pytest.fixture(scope="module")
def episode():
    settings = exo.quick(exo.PASSES[exo.FINAL_PASS])
    return exo.make_episode(settings, 0, 8, n_slots=12)


def test_episode_matches_the_slot_problems(episode, scene):
    assert episode.feat.shape == (12, 8, 4, 10) and episode.tracks.shape == (12, 4, 3)
    los = aerial.load_aerial_los()
    rng = np.random.default_rng(0)
    profiles = rng.integers(3, size=(12, 8))
    out = episode.evaluate(profiles)
    for t in (0, 5, 11):
        terms = episode.problem(t).terms(profiles[t][None, :])
        for key in ("utility", "rate", "sens", "peb", "paired"):
            assert np.allclose(out[key][t], terms[key][0], rtol=1e-12, atol=1e-13)
        # the sensing statistics of a slot are masked by the line of sight of that slot
        seen = episode.problem(t).sees
        assert seen.shape == (8, 4)
        assert los.is_free(episode.tracks[t]).all()
    assert out["switches"][0] == 0
    assert out["switches"][1] == (profiles[1] != profiles[0]).sum()
    assert np.array_equal(op.delayed(profiles, 0), profiles)
    assert np.array_equal(op.delayed(profiles, 3)[5], profiles[2])
    assert np.array_equal(op.delayed(profiles, 3)[:4], np.repeat(profiles[:1], 4, axis=0))
    assert op.command_messages(profiles)[1] == out["switches"][1]
    assert out["peb"].max() <= aerial.prior_error_bound(episode.eps) + 1e-9


def test_one_small_episode_is_deterministic(episode):
    def run_all():
        ref = op.run_reference(episode, np.random.default_rng([1, 2]), 1)
        init = ref["solutions"][0]
        return {
            "ref": ref["solutions"],
            "dbr": op.run_distributed(episode, init)["profiles"],
            "dbr1": op.run_distributed(episode, init, 1)["profiles"],
            "clustered": op.run_distributed(episode, init, 2, cluster=4)["profiles"],
            "pair": op.run_distributed(episode, init, pair=True)["profiles"],
            "sa": op.run_annealing(episode, init, 20, np.random.default_rng([3]), temperature=1e-3)[
                "solutions"
            ],
            "greedy": op.run_greedy(episode, init)["solutions"],
        }

    first, second = run_all(), run_all()
    for name in first:
        assert np.array_equal(first[name], second[name]), name
    # the converged controller is in equilibrium in every slot, the reference is pair-stable
    for t in range(1, episode.n_slots):
        assert op.deviation_gain(episode.problem(t), first["dbr"][t]) <= op.TOL
    values = episode.evaluate(first["ref"])["utility"]
    assert (values >= episode.evaluate(first["dbr"])["utility"] - 1e-9).all()


def test_distributed_controller_respects_the_subround_limit(episode):
    init = np.zeros(episode.n_agents, dtype=np.int64)
    one = op.run_distributed(episode, init, 1)
    assert (one["subrounds"][1:] == 1).all() and one["subrounds"][0] == 0
    # one sub-round moves agents of one colour class only
    changed = (one["profiles"][1:] != one["profiles"][:-1]).sum(axis=1)
    for t in range(1, episode.n_slots):
        colours = op.greedy_colouring(episode.problem(t).interaction_graph())
        assert changed[t - 1] <= np.bincount(colours).max()
    full = op.run_distributed(episode, init)
    assert (full["evals"][1:] >= one["evals"][1:]).all()


def test_experiment_tasks_are_deterministic():
    settings = exo.quick(exo.PASSES[exo.FINAL_PASS])
    assert exo._e12_task((settings, 0, 1, 0)) == exo._e12_task((settings, 0, 1, 0))
    a, b = exo._e9_task((settings, 1, 6)), exo._e9_task((settings, 1, 6))
    assert a == b
    deployed = exo.deploy(settings, 0, 8)
    assert np.array_equal(deployed["sites"], exo.deploy(settings, 0, 8)["sites"])
    assert len(np.unique(deployed["sites"])) == 8 and deployed["s_ref"] > 0
