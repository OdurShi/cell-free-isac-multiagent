"""Structure estimators, scenario generation, statistical channels and statistics."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_problem

from cfisac_ma import scenario as sc
from cfisac_ma.stats import mean_ci, t_quantile
from cfisac_ma.structure import estimate_structure, set_functions


def _sizes(sets):
    return (sets >= 0).sum(axis=1).astype(float)


def test_structure_of_reference_functions():
    rng = np.random.default_rng(0)
    modular = estimate_structure(
        lambda s: _sizes(s) / 10, 30, rng, n_samples=2000, max_base=4, max_extra=4
    )
    assert modular.violation_rate == 0.0
    assert modular.submodularity_ratio_min == pytest.approx(1.0)
    assert modular.dr_ratio_min == pytest.approx(1.0)
    assert modular.curvature_max == pytest.approx(0.0, abs=1e-9)
    assert modular.equilibrium_factor == pytest.approx(0.5)
    concave = estimate_structure(
        lambda s: np.sqrt(_sizes(s)), 30, rng, n_samples=2000, max_base=4, max_extra=4
    )
    assert concave.violation_rate == 0.0 and concave.submodularity_ratio_min >= 1.0
    assert 0.0 < concave.curvature_max < 1.0
    convex = estimate_structure(
        lambda s: _sizes(s) ** 2 / 100, 30, rng, n_samples=2000, max_base=4, max_extra=4
    )
    assert convex.violation_rate == 1.0
    assert convex.submodularity_ratio_min < 1.0 and convex.dr_ratio_min < 1.0
    assert convex.equilibrium_factor < 0.5


def test_structure_of_the_objective_parts():
    p = random_problem(0, n_sites=30, n_users=8, n_points=4)
    functions = set_functions(p, rx_sites=np.array([28, 29]))
    assert functions["sensing_split_tx"][1] == 28
    est = {
        name: estimate_structure(
            f, n, np.random.default_rng(1), n_samples=3000, max_base=4, max_extra=4
        )
        for name, (f, n) in functions.items()
    }
    for name in ("sum_rate", "sensing_split_tx"):
        assert est[name].violation_rate == 0.0, name
        assert est[name].submodularity_ratio_min >= 1.0 - 1e-9, name
    assert est["sensing_joint"].violation_rate > 0.0
    for e in est.values():
        assert e.monotonicity_violation_rate == 0.0


def test_users_are_servable_and_reproducible(scene):
    a = sc.make_scenario(scene, 5, n_users=40)
    b = sc.make_scenario(scene, 5, n_users=40)
    assert np.array_equal(a.users, b.users)
    assert not np.array_equal(a.users, sc.make_scenario(scene, 6, n_users=40).users)
    assert (np.abs(a.users) <= 250.0).all()
    snr = sc.user_snr(scene, a.users)
    assert (snr.max(axis=0) > 0).all()  # every user is reached by at least one valid site
    row, col = scene.cell_of(a.users)
    assert scene.outdoor[row, col].all()


def test_targets_follow_the_simulator_trajectories():
    t = sc.target_points(16)
    assert t.shape == (32, 3)
    circle, line = t[:16], t[16:]
    assert np.allclose(np.hypot(circle[:, 0] + 100, circle[:, 1] + 100), 80.0)
    assert np.allclose(circle[:, 2], 40.0) and np.allclose(line[:, 2], 20.0)
    assert np.allclose(line[:, 0], line[:, 1]) and (np.abs(line[:, 0]) <= 200).all()
    assert len(np.unique(np.round(line[:, 0], 6))) == 16


def test_sensing_geometry_matches_the_simulator_formula():
    pos = np.array([[0.0, 0.0, 10.0], [30.0, 40.0, 40.0]])
    tgt = np.array([[30.0, 40.0, 40.0]])
    w, u = sc.sensing_geometry(pos, tgt)
    d = np.sqrt(30.0**2 + 40.0**2 + 30.0**2)
    assert w[0, 0] == pytest.approx(1e10 * (3e8 / (4 * np.pi * d * 28e9)) ** 2)
    assert np.allclose(u[0, 0], np.array([30.0, 40.0, 30.0]) / d)
    assert w[1, 0] == pytest.approx(1e10 * (3e8 / (4 * np.pi * 1.0 * 28e9)) ** 2)  # clipped at 1 m
    assert np.allclose(u[1, 0], 0.0)


def test_hotspot_redraw_keeps_selected_hotspots(scene):
    rng = np.random.default_rng(0)
    new = sc.draw_hotspots(scene, rng, keep=(0, 2))
    assert new[0] == sc.SIMULATOR_HOTSPOTS[0] and new[2] == sc.SIMULATOR_HOTSPOTS[2]
    assert new[1].mean != sc.SIMULATOR_HOTSPOTS[1].mean
    assert new[1].std == sc.SIMULATOR_HOTSPOTS[1].std


def test_statistical_channels(scene):
    fit = sc.fit_log_distance(scene)
    assert 1.5 < fit["exponent"] < 3.0
    assert fit["rmse_db"] <= fit["free_space_rmse_db"]
    assert fit["free_intercept_fit"]["rmse_db"] <= fit["rmse_db"] + 1e-9
    scenario = sc.make_scenario(scene, 0)
    fs = sc.model_snr(scene, scenario.users, "free_space")
    ld = sc.model_snr(scene, scenario.users, "log_distance", exponent=2.0)
    assert np.allclose(fs, ld)
    # free space: gain depends on distance only and is cut at -110 dB (about 269 m)
    centres = sc.cell_centres()[scene.cell_of(scenario.users)]
    dist = np.linalg.norm(scene.positions[:, None, :] - centres[None, :, :], axis=-1)
    assert ((fs > 0) == (dist < 10 ** ((110.0 - 61.38493281289306) / 20.0))).all()
    problem = sc.make_problem(scene, scenario, channel="free_space")
    assert problem.utility(np.array([[1, 2, 3]]))[0] > 0


def test_student_t_interval():
    assert t_quantile(0.975, 9) == pytest.approx(2.262157, abs=1e-5)
    assert t_quantile(0.975, 1) == pytest.approx(12.7062, abs=1e-3)
    ci = mean_ci([1.0, 2.0, 3.0, 4.0, np.nan])
    assert ci["n"] == 4 and ci["mean"] == pytest.approx(2.5)
    assert ci["half"] == pytest.approx(3.182446 * np.std([1, 2, 3, 4], ddof=1) / 2, rel=1e-5)
    assert mean_ci([5.0])["half"] == 0.0
