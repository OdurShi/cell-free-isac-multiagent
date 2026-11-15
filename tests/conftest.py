"""Shared fixtures: the shipped scene and small synthetic problems."""

from __future__ import annotations

import numpy as np
import pytest

from cfisac_ma.data import load_scene
from cfisac_ma.objective import Problem


@pytest.fixture(scope="session")
def scene():
    return load_scene()


def random_problem(
    seed: int, n_sites: int = 8, n_users: int = 5, n_points: int = 3, **kw
) -> Problem:
    """Synthetic problem: sparse random user SNRs, random gains and directions."""
    rng = np.random.default_rng(seed)
    snr = rng.exponential(200.0, size=(n_sites, n_users)) * (rng.random((n_sites, n_users)) < 0.5)
    w = rng.uniform(0.05, 5.0, size=(n_sites, n_points))
    u = rng.normal(size=(n_sites, n_points, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    positions = rng.uniform(-100, 100, size=(n_sites, 3))
    opts = {"lam": 0.5, "eps": 1e-2}
    opts.update(kw)
    return Problem(snr, w, u, positions=positions, **opts)


@pytest.fixture
def make_random_problem():
    return random_problem


def random_mode_problem(
    seed: int,
    n_agents: int = 6,
    n_users: int = 5,
    n_targets: int = 3,
    *,
    reach: float = 0.5,
    sees: float = 0.7,
    **kw,
):
    """Synthetic mode-selection game: sparse user SNRs, random gains, directions and
    line-of-sight masks (``reach`` / ``sees`` are the probabilities of a link)."""
    from cfisac_ma.operation import N_FEATURES, ModeProblem

    rng = np.random.default_rng(seed)
    snr = rng.exponential(200.0, size=(n_agents, n_users))
    snr *= rng.random((n_agents, n_users)) < reach
    w = rng.uniform(0.05, 5.0, size=(n_agents, n_targets))
    w *= rng.random((n_agents, n_targets)) < sees
    u = rng.normal(size=(n_agents, n_targets, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    feat = np.empty((n_agents, n_targets, N_FEATURES))
    feat[..., 0] = w
    feat[..., 1:4] = w[..., None] * u
    for j, (c, d) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        feat[..., 4 + j] = w * u[..., c] * u[..., d]
    opts = {"lam": 0.5, "eps": 1e-2, "c_tx": 0.01, "c_rx": 0.005}
    opts.update(kw)
    problem = ModeProblem(snr, feat, **opts)
    problem.w, problem.u = w, u  # kept for the loop references of the tests
    return problem
