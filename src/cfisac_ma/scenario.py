"""Users, targets and the construction of a placement problem on the ray-traced scene.

The user hotspots, the two target trajectories, the carrier frequency, the free-space
sensing gain and the constants ``rho = 1e12`` and ``1e20`` are those of the simulator
(``simulation/isac_env.py`` of this repository).  Differences from the simulator are listed
in ``docs/notes.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cfisac_ma.data import (
    AREA_MAX_M,
    AREA_MIN_M,
    BLOCKAGE_THRESHOLD_DB,
    CELL_SIZE_M,
    GRID_SIZE,
    Scene,
    cell_index,
)
from cfisac_ma.objective import Problem

SPEED_OF_LIGHT = 3.0e8
CARRIER_FREQUENCY_HZ = 28.0e9
#: The simulator multiplies the product of two free-space gains by 1e20; we scale each gain
#: by 1e10, which is the same thing.
SENSING_GAIN_SCALE = 1.0e10
#: The simulator clips AP-target distances at 1 m in the gain and at 1e-3 m in the direction.
MIN_GAIN_DISTANCE_M = 1.0
MIN_DIRECTION_DISTANCE_M = 1.0e-3


@dataclass(frozen=True)
class Hotspot:
    """Isotropic Gaussian user hotspot on the ground plane."""

    mean: tuple[float, float]
    std: float
    weight: float


#: Hotspot mixture of the simulator (covariances 900, 400 and 625 m^2; weights 0.4, 0.4, 0.2).
SIMULATOR_HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot((-100.0, -100.0), 30.0, 0.4),
    Hotspot((150.0, 50.0), 20.0, 0.4),
    Hotspot((0.0, 200.0), 25.0, 0.2),
)


def servable_mask(scene: Scene) -> np.ndarray:
    """``[125, 125]`` cells that are outdoor and reached by at least one valid site."""
    return scene.outdoor & (scene.gain > 0.0).any(axis=0)


def draw_users(
    scene: Scene,
    rng: np.random.Generator,
    n_users: int,
    hotspots: tuple[Hotspot, ...] = SIMULATOR_HOTSPOTS,
    max_tries: int = 100_000,
) -> np.ndarray:
    """Draw ``n_users`` ground positions ``[K, 2]`` from a hotspot mixture.

    A hotspot is chosen with its weight; positions are then drawn from that hotspot until one
    falls inside the area, on an outdoor cell (simulator rule) that at least one valid
    candidate site reaches.
    """
    weights = np.array([h.weight for h in hotspots], dtype=np.float64)
    weights = weights / weights.sum()
    ok = servable_mask(scene)
    users = np.empty((n_users, 2))
    for k in range(n_users):
        h = hotspots[int(rng.choice(len(hotspots), p=weights))]
        for _ in range(max_tries):
            xy = rng.normal(h.mean, h.std)
            if not (AREA_MIN_M <= xy[0] <= AREA_MAX_M and AREA_MIN_M <= xy[1] <= AREA_MAX_M):
                continue
            row, col = cell_index(xy)
            if ok[row, col]:
                users[k] = xy
                break
        else:
            raise RuntimeError(f"no servable position found for hotspot at {h.mean}")
    return users


def draw_hotspots(
    scene: Scene,
    rng: np.random.Generator,
    template: tuple[Hotspot, ...] = SIMULATOR_HOTSPOTS,
    keep: tuple[int, ...] = (),
    margin: float = 50.0,
) -> tuple[Hotspot, ...]:
    """New hotspot centres, uniform over servable cells at least ``margin`` from the border.

    Standard deviations and weights are those of ``template``; hotspots listed in ``keep``
    are not moved.
    """
    ok = servable_mask(scene)
    out = []
    for i, h in enumerate(template):
        if i in keep:
            out.append(h)
            continue
        while True:
            xy = rng.uniform(AREA_MIN_M + margin, AREA_MAX_M - margin, size=2)
            row, col = cell_index(xy)
            if ok[row, col]:
                break
        out.append(Hotspot((float(xy[0]), float(xy[1])), h.std, h.weight))
    return tuple(out)


def target_points(n_per_trajectory: int = 16) -> np.ndarray:
    """Sample points ``[2 * Q, 3]`` of the simulator's two aerial target trajectories.

    * Circle: centre (-100, -100), radius 80 m, height 40 m, angles ``2 pi (j + 1/2) / Q``.
    * Line: from (200, 200) to (-200, -200) at height 20 m with the simulator's sinusoidal
      progress ``(sin(theta) + 1) / 2``; ``theta`` takes the ``Q`` mid-points of a regular
      partition of ``[-pi/2, pi/2]``, which samples the stationary distribution of the
      motion (denser near the two ends).
    """
    q = n_per_trajectory
    j = np.arange(q) + 0.5
    angle = 2.0 * math.pi * j / q
    circle = np.stack(
        [-100.0 + 80.0 * np.cos(angle), -100.0 + 80.0 * np.sin(angle), np.full(q, 40.0)], axis=1
    )
    progress = (np.sin(-math.pi / 2 + math.pi * j / q) + 1.0) / 2.0
    line = np.stack([200.0 - 400.0 * progress, 200.0 - 400.0 * progress, np.full(q, 20.0)], axis=1)
    return np.concatenate([circle, line], axis=0)


def user_snr(scene: Scene, users: np.ndarray, rho: float = 1.0e12) -> np.ndarray:
    """``[V, K]`` values ``rho * beta_{a,k}`` (0 where the path is blocked)."""
    row, col = cell_index(users)
    return rho * scene.gain[:, row, col]


def sensing_geometry(positions: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Free-space sensing gain and AP-to-target unit vectors.

    Returns:
        ``w``: ``[V, P]`` scaled gain ``1e10 * (c / (4 pi d f))**2`` with ``d`` clipped at 1 m.
        ``u``: ``[V, P, 3]`` unit vectors from each AP to each target point.
    """
    diff = targets[None, :, :] - positions[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    wavelength_term = SPEED_OF_LIGHT / (4.0 * math.pi * CARRIER_FREQUENCY_HZ)
    w = SENSING_GAIN_SCALE * (wavelength_term / np.maximum(dist, MIN_GAIN_DISTANCE_M)) ** 2
    u = diff / np.maximum(dist, MIN_DIRECTION_DISTANCE_M)[..., None]
    return w, u


@dataclass(frozen=True)
class Scenario:
    """One random scenario: where the users and the target sample points are."""

    users: np.ndarray
    targets: np.ndarray
    hotspots: tuple[Hotspot, ...]
    seed: int | None = None


def make_scenario(
    scene: Scene,
    seed: int,
    n_users: int = 15,
    hotspots: tuple[Hotspot, ...] = SIMULATOR_HOTSPOTS,
    n_target_points: int = 16,
) -> Scenario:
    """Scenario with users drawn with ``numpy.random.default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    users = draw_users(scene, rng, n_users, hotspots)
    return Scenario(
        users=users, targets=target_points(n_target_points), hotspots=hotspots, seed=seed
    )


def make_problem(
    scene: Scene,
    scenario: Scenario,
    *,
    lam: float = 0.5,
    eps: float = 1.0e-4,
    rho: float = 1.0e12,
    rate: str = "sum",
    sensing: str = "joint",
    n_tx: int | None = None,
    sites: np.ndarray | None = None,
    channel: str = "raytraced",
    exponent: float = 2.0,
) -> Problem:
    """Placement problem of a scenario.

    Args:
        channel: ``"raytraced"`` (the shipped database), ``"free_space"`` or
            ``"log_distance"`` (statistical channels for the user links, see
            :func:`model_snr`); the sensing term is the same in all three.
        exponent: path-loss exponent of the ``"log_distance"`` channel.
        lam: weight of the communication term in the team utility.
        eps: regulariser of the D-optimality criterion, in the scaled units of the sensing
            information matrix.
        rho: transmit-power-to-noise ratio in the user rate.
        rate: ``"sum"`` (sum rate) or ``"maxmin"`` (minimum user rate).
        sensing: ``"joint"`` (every AP transmits and receives) or ``"split"`` (the first
            ``n_tx`` agents transmit, the others receive).
        sites: optional subset of valid-site indices; the problem is then restricted to
            these candidates (indices in the returned problem refer to this subset).
    """
    if channel == "raytraced":
        snr = user_snr(scene, scenario.users, rho)
    else:
        snr = model_snr(scene, scenario.users, channel, exponent, rho)
    w, u = sensing_geometry(scene.positions, scenario.targets)
    positions = scene.positions
    if sites is not None:
        sites = np.asarray(sites)
        snr, w, u, positions = snr[sites], w[sites], u[sites], positions[sites]
    return Problem(
        snr=snr,
        w=w,
        u=u,
        lam=lam,
        eps=eps,
        rate=rate,
        sensing=sensing,
        n_tx=n_tx,
        positions=positions,
    )


# ---------------------------------------------------------------------------- statistical channels
CHANNEL_MODELS = ("raytraced", "free_space", "log_distance")
USER_HEIGHT_M = 1.5


def free_space_gain_db(distance_m: np.ndarray) -> np.ndarray:
    """Friis free-space gain in dB at the carrier frequency (isotropic antennas)."""
    k = SPEED_OF_LIGHT / (4.0 * math.pi * CARRIER_FREQUENCY_HZ)
    return 20.0 * np.log10(k / np.maximum(distance_m, MIN_GAIN_DISTANCE_M))


def cell_centres() -> np.ndarray:
    """``[125, 125, 3]`` coordinates of the ground-grid cell centres (row = y, column = x)."""
    c = AREA_MIN_M + CELL_SIZE_M * (np.arange(GRID_SIZE) + 0.5)
    x, y = np.meshgrid(c, c)
    return np.stack([x, y, np.full_like(x, USER_HEIGHT_M)], axis=-1)


def site_cell_distances(scene: Scene) -> np.ndarray:
    """``[V, 125, 125]`` 3-D distance from every valid site to every cell centre."""
    centres = cell_centres()
    diff = scene.positions[:, None, None, :] - centres[None, :, :, :]
    return np.linalg.norm(diff, axis=-1)


def fit_log_distance(scene: Scene) -> dict:
    """Least-squares fit of a log-distance model to the unblocked ray-traced gains.

    Model (close-in reference at 1 m): ``G_dB(d) = G_fs(1 m) - 10 n log10(d)`` with the
    free-space gain at 1 m as intercept and the single fitted exponent ``n``.  Only the
    AP-to-cell distances enter; blocked entries (below -110 dB) are not part of the fit.

    Returns a dict with the exponent, the root-mean-square error in dB of this model, of
    free space (``n = 2``) and of a two-parameter fit with a free intercept (reported for
    reference only), and the number of fitted entries.
    """
    dist = site_cell_distances(scene)
    ok = scene.gain > 0.0
    x = np.log10(np.maximum(dist[ok], MIN_GAIN_DISTANCE_M))
    y = 10.0 * np.log10(scene.gain[ok])
    g0 = float(free_space_gain_db(np.array(1.0)))
    exponent = float(-np.sum((y - g0) * x) / (10.0 * np.sum(x * x)))
    rmse = float(np.sqrt(np.mean((y - (g0 - 10.0 * exponent * x)) ** 2)))
    rmse_fs = float(np.sqrt(np.mean((y - (g0 - 20.0 * x)) ** 2)))
    slope, intercept = np.polyfit(x, y, 1)
    rmse_free = float(np.sqrt(np.mean((y - (intercept + slope * x)) ** 2)))
    return {
        "exponent": exponent,
        "intercept_db": g0,
        "rmse_db": rmse,
        "free_space_rmse_db": rmse_fs,
        "free_intercept_fit": {
            "exponent": float(-slope / 10.0),
            "intercept_db": float(intercept),
            "rmse_db": rmse_free,
        },
        "n_entries": int(ok.sum()),
        "unblocked_fraction": float(ok.mean()),
    }


def model_snr(
    scene: Scene, users: np.ndarray, model: str, exponent: float = 2.0, rho: float = 1.0e12
) -> np.ndarray:
    """``[V, K]`` values ``rho * beta`` under a statistical channel that ignores the scene.

    ``model="free_space"`` uses the Friis gain, ``model="log_distance"`` the close-in
    log-distance model with the given exponent.  The distance is taken from the site to the
    centre of the user's grid cell; gains below -110 dB are set to 0, as for the ray-traced
    channel.
    """
    if model == "free_space":
        exponent = 2.0
    elif model != "log_distance":
        raise ValueError("model must be 'free_space' or 'log_distance'")
    row, col = cell_index(users)
    centres = cell_centres()[row, col]
    dist = np.linalg.norm(scene.positions[:, None, :] - centres[None, :, :], axis=-1)
    g0 = float(free_space_gain_db(np.array(1.0)))
    gain_db = g0 - 10.0 * exponent * np.log10(np.maximum(dist, MIN_GAIN_DISTANCE_M))
    return np.where(gain_db >= BLOCKAGE_THRESHOLD_DB, rho * 10.0 ** (gain_db / 10.0), 0.0)
