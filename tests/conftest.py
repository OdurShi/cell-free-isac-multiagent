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
