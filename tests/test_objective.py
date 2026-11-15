"""The objective against loop-based references transcribed from the simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import random_problem

from cfisac_ma.objective import Evaluator, first_occurrence_mask, trace_value_at
from cfisac_ma.scenario import make_problem, make_scenario, sensing_geometry, user_snr


def simulator_rates(scene, users, selected):
    """Transcription of ``calculate_communication_rate`` for outdoor users."""
    threshold = 10 ** (-110.0 / 10.0)
    sum_rate, min_rate = 0.0, float("inf")
    for x, y in users:
        col = int(np.clip((x + 250.0) / 4.0, 0, 124))
        row = int(np.clip((y + 250.0) / 4.0, 0, 124))
        snr = 0.0
        for a in selected:
            g = scene.gain[a, row, col]
            snr += 0.0 if g < threshold * (1 - 1e-6) else g
        rate = math.log2(1 + snr * 1e12)
        sum_rate += rate
        min_rate = min(min_rate, rate)
    return sum_rate, min_rate


def simulator_fim_det(positions, tx, rx, target):
    """Transcription of ``_calculate_single_fim_det`` (without the final clipping)."""
    c, f = 3e8, 28e9
    tp = np.asarray(target, dtype=float)
    fim = np.zeros((3, 3))
    for m in tx:
        dm = max(float(np.linalg.norm(tp - positions[m])), 1.0)
        beta_m = (c / (4 * math.pi * dm * f)) ** 2
        um = (tp - positions[m]) / max(float(np.linalg.norm(tp - positions[m])), 1e-3)
        for n in rx:
            dn = max(float(np.linalg.norm(tp - positions[n])), 1.0)
            beta_n = (c / (4 * math.pi * dn * f)) ** 2
            un = (tp - positions[n]) / max(float(np.linalg.norm(tp - positions[n])), 1e-3)
            s = um + un
            fim += beta_m * beta_n * 1e20 * np.outer(s, s)
    f11, f22, f33 = fim[0, 0], fim[1, 1], fim[2, 2]
    f12, f13, f23 = fim[0, 1], fim[0, 2], fim[1, 2]
    det = f11 * (f22 * f33 - f23**2) - f12 * (f12 * f33 - f13 * f23) + f13 * (f12 * f23 - f13 * f22)
    return det, fim


@pytest.fixture(scope="module")
def real_problem(scene):
    return make_problem(scene, make_scenario(scene, 3, n_users=15))


def test_rates_match_simulator_loops(scene, real_problem):
    sc = make_scenario(scene, 3, n_users=15)
    rng = np.random.default_rng(0)
    for _ in range(5):
        sel = rng.choice(scene.n_sites, size=6, replace=False)
        comp = real_problem.components(sel[None, :])
        ref_sum, ref_min = simulator_rates(scene, sc.users, sel)
        assert comp.rate_sum[0] == pytest.approx(ref_sum, rel=1e-12)
        assert comp.rate_min[0] == pytest.approx(ref_min, rel=1e-12, abs=1e-12)


def test_split_fim_matches_simulator_loops(scene):
    sc = make_scenario(scene, 3, n_users=15)
    split = make_problem(scene, sc, sensing="split", n_tx=2)
    rng = np.random.default_rng(1)
    for _ in range(4):
        sel = rng.choice(scene.n_sites, size=5, replace=False)
        fim = split.fim(sel[None, :])[0]
        for p in (0, 7, 20, 31):
            det, ref = simulator_fim_det(scene.positions, sel[:2], sel[2:], sc.targets[p])
            assert np.allclose(fim[p], ref, rtol=1e-10, atol=1e-12)
            assert np.linalg.det(fim[p]) == pytest.approx(det, rel=1e-7, abs=1e-9)


def test_joint_fim_is_the_double_sum(scene, real_problem):
    sc = make_scenario(scene, 3, n_users=15)
    w, u = sensing_geometry(scene.positions, sc.targets)
    sel = np.array([5, 77, 140, 300])
    fim = real_problem.fim(sel[None, :])[0]
    for p in (0, 13, 25):
        ref = np.zeros((3, 3))
        for m in sel:
            for n in sel:
                s = u[m, p] + u[n, p]
                ref += w[m, p] * w[n, p] * np.outer(s, s)
        assert np.allclose(fim[p], ref, rtol=1e-10)
    # joint roles equal split roles with the same set on both sides
    split = real_problem.with_options(sensing="split", n_tx=4)
    both = np.concatenate([sel, sel])
    # (duplicates across roles are dropped by the placement rule, so compare matrices directly)
    ft = real_problem._table[sel].sum(axis=0)[real_problem.n_users :]
    from cfisac_ma.objective import _joint_entries, _split_entries

    f = ft.reshape(real_problem.n_points, 10)
    for a, b in zip(_joint_entries(f), _split_entries(f, f), strict=True):
        assert np.allclose(a, b, rtol=1e-12)
    assert split.n_tx == 4 and len(both) == 8


def test_logdet_matches_eigenvalues(real_problem):
    rng = np.random.default_rng(2)
    for size in (1, 2, 3, 6, 16):
        sets = np.stack([rng.choice(real_problem.n_sites, size, replace=False) for _ in range(40)])
        fim = real_problem.fim(sets)
        eig = np.linalg.eigvalsh(fim + real_problem.eps * np.eye(3))
        ref = np.log(eig).sum(axis=-1).mean(axis=-1)
        assert np.allclose(real_problem.components(sets).logdet, ref, rtol=1e-7, atol=1e-7)


def test_normalisation(real_problem):
    p = real_problem
    empty = np.full((1, 4), -1)
    assert p.utility(empty)[0] == pytest.approx(0.0, abs=1e-12)
    everything = np.arange(p.n_sites)[None, :]
    assert p.utility(everything)[0] == pytest.approx(1.0, rel=1e-12)
    comm, sens = p.normalised(p.components(np.array([[3, 50, 200, 310]])))
    assert 0.0 <= comm[0] <= 1.0 and 0.0 <= sens[0] <= 1.0
    for lam in (0.0, 0.3, 1.0):
        q = p.with_options(lam=lam)
        assert q.utility(np.array([[3, 50, 200, 310]]))[0] == pytest.approx(
            lam * comm[0] + (1 - lam) * sens[0], rel=1e-12
        )


def test_duplicates_and_absent_agents(real_problem):
    p = real_problem
    a = p.utility(np.array([[10, 20, 30]]))[0]
    assert p.utility(np.array([[10, 20, 30, 20, -1, 10]]))[0] == pytest.approx(a, rel=1e-12)
    assert p.utility(np.array([[30, -1, 10, 20]]))[0] == pytest.approx(a, rel=1e-12)
    mask = first_occurrence_mask(np.array([[4, 4, -1, 7, 4]]))
    assert mask.tolist() == [[1.0, 0.0, 0.0, 1.0, 0.0]]


@pytest.mark.parametrize("mode", ["joint", "split"])
def test_utility_replace_equals_batch(mode):
    kw = {"sensing": "split", "n_tx": 2} if mode == "split" else {}
    p = random_problem(5, n_sites=12, **kw)
    placement = np.array([3, 7, 7, 1, -1])
    for agent in range(len(placement)):
        batch = np.repeat(placement[None, :], p.n_sites, axis=0)
        batch[:, agent] = np.arange(p.n_sites)
        assert np.allclose(p.utility_replace(placement, agent), p.utility(batch), rtol=1e-12)
    cand = np.array([0, 3, 9])
    assert np.allclose(p.utility_replace(placement, 0, cand), p.utility_replace(placement, 0)[cand])


def test_monotone_in_the_set():
    for seed in range(5):
        p = random_problem(seed, n_sites=10)
        rng = np.random.default_rng(seed)
        perm = np.stack([rng.permutation(10) for _ in range(50)])
        prev = np.zeros(50)
        for size in range(1, 11):
            sets = np.where(np.arange(10)[None, :] < size, perm, -1)
            value = p.utility(sets)
            assert (value >= prev - 1e-12).all()
            prev = value


def test_evaluator_counts_and_traces():
    p = random_problem(7, n_sites=12)
    ev = Evaluator(p, budget=40)
    placement = np.array([0, 1, 2])
    v = ev.utility(placement[None, :])
    assert ev.n_evals == 1 and ev.best_value == v[0]
    values = ev.replace(placement, 1)
    assert ev.n_evals == 13
    assert ev.best_value == pytest.approx(max(v[0], values.max()))
    assert p.utility(ev.best_placement[None, :])[0] == pytest.approx(ev.best_value)
    w = ev.without(placement, 2)
    assert ev.n_evals == 14 and w <= v[0] + 1e-12
    steps = [t[0] for t in ev.trace]
    assert steps == sorted(steps) and [t[1] for t in ev.trace] == sorted(t[1] for t in ev.trace)
    assert trace_value_at(ev.trace, [0])[0] != trace_value_at(ev.trace, [0])[0]  # nan before start
    assert trace_value_at(ev.trace, [14])[0] == pytest.approx(ev.best_value)
    with pytest.raises(RuntimeError):
        ev.utility(np.zeros((27, 3), dtype=int))


def test_user_snr_uses_the_cell_of_the_user(scene):
    users = np.array([[-100.0, -100.0], [12.3, 45.6]])
    snr = user_snr(scene, users)
    assert snr.shape == (scene.n_sites, 2)
    assert snr[17, 0] == 1e12 * scene.gain[17, 37, 37]
