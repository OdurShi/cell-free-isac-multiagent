"""The package against the simulator itself (``simulation/isac_env.py``).

``tests/test_objective.py`` compares the objective with loop transcriptions of the
simulator's functions.  The tests below call the simulator's own methods on the simulator's
own arrays, so they tie the two parts of the repository together: same candidate sites, same
user rates, same sensing information, same target trajectories.  Like
``tests/test_isac_env.py`` they need the classic ``gym`` package and are skipped without it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gym")

SIMULATION_DIR = Path(__file__).resolve().parents[1] / "simulation"
sys.path.insert(0, str(SIMULATION_DIR))

from isac_env import CellFreeISAC3DEnv  # noqa: E402

from cfisac_ma.scenario import (  # noqa: E402
    SIMULATOR_HOTSPOTS,
    Scenario,
    make_problem,
    make_scenario,
    target_points,
)


@pytest.fixture(scope="module")
def env() -> CellFreeISAC3DEnv:
    return CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type="max-sum")


def test_valid_candidate_sites_are_the_simulators(scene, env):
    assert env.num_candidates == scene.n_sites == 345
    assert np.array_equal(env.valid_indices, scene.site_ids)
    assert np.array_equal(env.candidate_positions[env.valid_indices], scene.positions)


def test_user_rates_match_the_simulator(scene, env):
    """Sum and minimum rate of the simulator's own outdoor users.

    Indoor users get a penetration loss that the simulator redraws at every evaluation, so
    only the outdoor users have a deterministic rate.  The compact database is quantised
    (largest relative error of a gain 6.63e-05), hence the tolerance.
    """
    outdoor = [u for u in env.users_info if not u["is_indoor"]]
    assert len(outdoor) >= 5
    users = np.array([u["pos"][:2] for u in outdoor], dtype=np.float64)
    problem = make_problem(scene, Scenario(users, target_points(), SIMULATOR_HOTSPOTS))

    all_users, n_users = env.users_info, env.K
    env.users_info, env.K = outdoor, len(outdoor)
    try:
        rng = np.random.default_rng(0)
        for _ in range(8):
            sel = rng.choice(scene.n_sites, size=6, replace=False)
            ref_sum, ref_min = env.calculate_communication_rate(scene.site_ids[sel])
            comp = problem.components(sel[None, :])
            assert comp.rate_sum[0] == pytest.approx(ref_sum, rel=1e-4)
            assert comp.rate_min[0] == pytest.approx(ref_min, rel=1e-4, abs=1e-9)
    finally:
        env.users_info, env.K = all_users, n_users


def test_split_role_information_matches_the_simulator(scene, env):
    """Determinant of the split-role information matrix at single target points."""
    scenario = make_scenario(scene, 3, n_users=15)
    split = make_problem(scene, scenario, sensing="split", n_tx=2)
    rng = np.random.default_rng(1)
    for _ in range(4):
        sel = rng.choice(scene.n_sites, size=5, replace=False)
        fim = split.fim(sel[None, :])[0]
        ids = scene.site_ids[sel]
        for p in (0, 7, 20, 31):
            ref = env._calculate_single_fim_det(ids[:2], ids[2:], scenario.targets[p])
            assert max(np.linalg.det(fim[p]), 1e-12) == pytest.approx(ref, rel=1e-7, abs=1e-9)


def test_target_sample_points_lie_on_the_simulators_trajectories(env):
    circle, line = env.targets_params
    assert (circle["type"], line["type"]) == ("circle", "line")
    points = target_points(16)
    on_circle, on_line = points[:16], points[16:]

    radius = np.hypot(on_circle[:, 0] - circle["center"][0], on_circle[:, 1] - circle["center"][1])
    assert np.allclose(radius, circle["radius"]) and np.allclose(on_circle[:, 2], circle["z"])

    start, end = np.array(line["start"], dtype=float), np.array(line["end"], dtype=float)
    progress = (on_line[:, :2] - start) @ (end - start) / ((end - start) @ (end - start))
    assert np.allclose(start + progress[:, None] * (end - start), on_line[:, :2])
    assert (progress > 0).all() and (progress < 1).all()
    assert np.allclose(on_line[:, 2], line["z"])
    # the progress values are those of the simulator's motion law at 16 phases
    phases = -math.pi / 2 + math.pi * (np.arange(16) + 0.5) / 16
    assert np.allclose(progress, (np.sin(phases) + 1.0) / 2.0)
