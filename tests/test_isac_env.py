"""Smoke tests of ``simulation/isac_env.py`` on the shipped channel databases.

They need only NumPy and the classic ``gym`` package (no Sionna, TensorFlow or GPU): the
environment reads the pre-computed path-gain database and the candidate positions that are
stored next to the scripts.
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

INFO_KEYS = {
    "selected_aps", "sum_rate", "min_rate", "fim_det", "norm_rate", "norm_fim", "norm_penalty",
}


@pytest.fixture(scope="module")
def env() -> CellFreeISAC3DEnv:
    return CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type="max-sum")


def test_shipped_databases_have_the_expected_layout():
    database = np.load(SIMULATION_DIR / "paris_pg_database_3d.npy", mmap_mode="r")
    positions = np.load(SIMULATION_DIR / "candidate_positions_3d.npy")
    assert database.shape == (11, 11, 7, 125, 125) and database.dtype == np.float32
    assert positions.shape == (11 * 11 * 7, 3)
    assert np.isfinite(positions).all()
    assert positions[:, :2].min() == -250.0 and positions[:, :2].max() == 250.0
    assert positions[:, 2].min() == 5.0 and positions[:, 2].max() == 35.0
    curve = np.load(SIMULATION_DIR / "drl_real_curve.npy")
    assert curve.shape == (2000,) and np.isfinite(curve).all()


def test_spaces_and_reset(env):
    assert env.num_candidates == len(env.valid_indices) > env.M + env.N
    assert env.observation_space.shape == (15 * 3 + 2 * 2,)
    assert env.action_space.shape == ((2 + 2) * 3,)
    state = env.reset()
    assert state.shape == env.observation_space.shape and state.dtype == np.float32
    assert np.isfinite(state).all()
    assert env.observation_space.contains(state)
    assert len(env.users_info) == 15 and len(env.targets_pos) == 2


def test_step_returns_finite_consistent_values(env):
    np.random.seed(0)
    env.reset()
    for _ in range(5):
        action = np.random.uniform(-1.0, 1.0, env.action_space.shape[0])
        state, reward, done, info = env.step(action)
        assert state.shape == env.observation_space.shape and np.isfinite(state).all()
        assert math.isfinite(reward) and not done
        assert INFO_KEYS <= set(info)
        selected = np.asarray(info["selected_aps"])
        assert selected.shape == (4,) and len(set(selected.tolist())) == 4
        assert set(selected.tolist()) <= set(env.valid_indices.tolist())
        assert info["sum_rate"] >= info["min_rate"] >= 0.0
        assert info["fim_det"] > 0.0 and math.isfinite(info["fim_det"])
        assert info["norm_penalty"] >= 0.0
        expected = info["norm_rate"] + info["norm_fim"] - info["norm_penalty"]
        assert reward == pytest.approx(expected)


def test_out_of_range_actions_are_clipped(env):
    env.reset()
    _, reward, _, info = env.step(np.full(env.action_space.shape[0], 7.0))
    assert math.isfinite(reward)
    assert len(set(np.asarray(info["selected_aps"]).tolist())) == 4


def test_episode_ends_after_max_steps(env):
    env.reset()
    action = np.zeros(env.action_space.shape[0])
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(action)
        steps += 1
        assert steps <= env.max_steps
    assert steps == env.max_steps == 200


def test_max_min_objective_and_fixed_user_layout():
    other = CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type="max-min")
    again = CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type="max-min")
    # The user layout is drawn with a fixed seed inside the constructor.
    assert [u["pos"] for u in other.users_info] == [u["pos"] for u in again.users_info]
    other.reset()
    _, reward, _, info = other.step(np.zeros(other.action_space.shape[0]))
    expected = max(info["min_rate"], 1e-5) / 2.0 + info["norm_fim"] - info["norm_penalty"]
    assert reward == pytest.approx(expected)
