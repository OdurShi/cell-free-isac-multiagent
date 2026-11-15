"""Aerial targets on the ray-traced scene: line-of-sight tables, planning points, UAV paths
and the physical scale of the sensing information.

``scripts/raytrace_scene.py`` tests, with Sionna RT on the ``etoile`` scene, whether the
straight segment from every valid candidate site to the centre of every cell of the
125 x 125 grid is free of geometry, at the flight levels 10, 20, 30 and 40 m and at the user
height.  The result is shipped as ``data/aerial_los.npz``; this module reads it and offers

* :class:`AerialLos` -- the tables and the lookup ``visible(points)`` used to mask the
  sensing links (a site contributes to the information about a target only if it sees it);
* :func:`planning_points` -- a seeded set of surveillance points on the free cells;
* :func:`waypoint_path`, :func:`target_tracks` -- random-waypoint UAV paths whose segments
  stay in free cells of their flight level;
* :func:`sensing_scale`, :func:`position_error_bound` -- the reading of the simulator's
  factor ``1e20`` as a bistatic-range Fisher information in 1/m^2 and the Bayesian position
  error bound in metres that follows from it (``docs/theory.md``, Section 10).
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfisac_ma.data import (
    AREA_MIN_M,
    CELL_SIZE_M,
    GRID_SIZE,
    cell_index,
    default_data_dir,
)
from cfisac_ma.scenario import CARRIER_FREQUENCY_HZ, SPEED_OF_LIGHT

LOS_FILE = "aerial_los.npz"
LOS_MANIFEST = "aerial_manifest.json"
N_CELLS = GRID_SIZE * GRID_SIZE

BOLTZMANN_J_PER_K = 1.380649e-23
REFERENCE_TEMPERATURE_K = 290.0


@dataclass(frozen=True)
class AerialLos:
    """Line-of-sight tables of the valid candidate sites.

    Attributes:
        levels_m: ``[L]`` flight levels (m).
        los: ``[L, V, 15625]`` bool; ``los[l, a, r * 125 + c]`` is true if site ``a`` sees
            the centre of cell ``(r, c)`` (row = y, column = x) at level ``l``.
        ground_los: ``[V, 15625]`` bool, the same at the user height of 1.5 m.
        free: ``[L, 125, 125]`` bool; the highest roof inside the cell is at least 2 m below
            the level, i.e. a UAV can be there.
        site_ids: ``[V]`` index of each valid site among the 847 candidate positions.
        site_roof_m: ``[V]`` height of the highest surface above each site's ground position.
    """

    levels_m: np.ndarray
    los: np.ndarray
    ground_los: np.ndarray
    free: np.ndarray
    site_ids: np.ndarray
    site_roof_m: np.ndarray

    @property
    def n_levels(self) -> int:
        return len(self.levels_m)

    def level_index(self, z: np.ndarray) -> np.ndarray:
        """Index of the flight level of each height ``z``; heights off the levels raise."""
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        index = np.abs(z[:, None] - self.levels_m[None, :]).argmin(axis=1)
        if not np.allclose(self.levels_m[index], z, atol=1e-6):
            raise ValueError(f"heights {np.unique(z)} are not flight levels {self.levels_m}")
        return index

    def cell_of(self, points: np.ndarray) -> np.ndarray:
        """Flat cell index ``row * 125 + col`` of points ``[..., >=2]``."""
        row, col = cell_index(np.asarray(points)[..., :2])
        return row * GRID_SIZE + col

    def visible(self, points: np.ndarray, sites: np.ndarray | None = None) -> np.ndarray:
        """``[V, P]`` (or ``[len(sites), P]``) bool: the site sees the point.

        ``points`` is ``[P, 3]``; the test is the table entry of the point's flight level
        and of the cell that contains the point (the tested position is the cell centre).
        """
        points = np.asarray(points, dtype=np.float64)
        level = self.level_index(points[:, 2])
        table = self.los[level, :, self.cell_of(points)]  # [P, V]
        table = table.T
        return table if sites is None else table[np.asarray(sites)]

    def is_free(self, points: np.ndarray) -> np.ndarray:
        """``[P]`` bool: the point lies in a free cell of its flight level."""
        points = np.asarray(points, dtype=np.float64)
        row, col = cell_index(points[:, :2])
        return self.free[self.level_index(points[:, 2]), row, col]


@functools.lru_cache(maxsize=2)
def _load(path: str) -> AerialLos:
    with np.load(path) as f:
        los = np.unpackbits(f["los_bits"], axis=-1, count=N_CELLS).astype(bool)
        ground = np.unpackbits(f["ground_los_bits"], axis=-1, count=N_CELLS).astype(bool)
        return AerialLos(
            levels_m=f["levels_m"].astype(np.float64),
            los=los,
            ground_los=ground,
            free=f["free"].astype(bool),
            site_ids=f["site_ids"].astype(np.int64),
            site_roof_m=f["site_roof_m"].astype(np.float64),
        )


def load_aerial_los(data_dir: str | os.PathLike | None = None) -> AerialLos:
    """Load ``data/aerial_los.npz`` (cached per process)."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    return _load(str(data_dir / LOS_FILE))


def cell_centre(flat: np.ndarray) -> np.ndarray:
    """``[n, 2]`` x, y of the centres of cells given by their flat index."""
    flat = np.asarray(flat)
    row, col = np.divmod(flat, GRID_SIZE)
    return np.stack(
        [AREA_MIN_M + (col + 0.5) * CELL_SIZE_M, AREA_MIN_M + (row + 0.5) * CELL_SIZE_M], axis=-1
    )


def planning_points(los: AerialLos, rng: np.random.Generator, n_per_level: int = 16) -> np.ndarray:
    """Surveillance points ``[L * n, 3]``: centres of free cells, uniform per flight level.

    Cells are drawn without replacement, the levels in ascending order.
    """
    out = []
    for li, z in enumerate(los.levels_m):
        cells = np.flatnonzero(los.free[li].ravel())
        pick = rng.choice(cells, size=n_per_level, replace=False)
        xy = cell_centre(pick)
        out.append(np.concatenate([xy, np.full((n_per_level, 1), z)], axis=1))
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------------------ UAV paths
def largest_free_component(free: np.ndarray) -> np.ndarray:
    """Mask of the largest 4-connected component of a boolean grid."""
    label = np.zeros(free.shape, dtype=np.int64)
    sizes = [0]
    n_rows, n_cols = free.shape
    for r0, c0 in zip(*np.nonzero(free), strict=True):
        if label[r0, c0]:
            continue
        current = len(sizes)
        label[r0, c0] = current
        stack = [(int(r0), int(c0))]
        count = 0
        while stack:
            r, c = stack.pop()
            count += 1
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < n_rows and 0 <= cc < n_cols and free[rr, cc] and not label[rr, cc]:
                    label[rr, cc] = current
                    stack.append((rr, cc))
        sizes.append(count)
    return label == int(np.argmax(sizes))


def segment_is_free(free: np.ndarray, a: np.ndarray, b: np.ndarray, step_m: float = 1.0) -> bool:
    """True if the segment from ``a`` to ``b`` (x, y) only crosses free cells.

    The segment is sampled every ``step_m`` (at most); with 4 m cells a step of 1 m can miss
    a cell only where the segment clips its corner by less than a metre.
    """
    n = max(int(math.ceil(float(np.linalg.norm(b - a)) / step_m)), 1) + 1
    t = np.linspace(0.0, 1.0, n)[:, None]
    row, col = cell_index(a[None, :] + t * (b - a)[None, :])
    return bool(free[row, col].all())


def waypoint_path(
    los: AerialLos,
    level: int,
    rng: np.random.Generator,
    length_m: float,
    max_tries: int = 20_000,
) -> np.ndarray:
    """Random-waypoint path ``[W, 2]`` of at least ``length_m`` at flight level ``level``.

    The start is the centre of a uniformly drawn cell of the largest connected free region
    of the level.  Each next waypoint is the centre of a uniformly drawn cell of that
    region, redrawn until the straight segment to it stays in free cells.
    """
    free = los.free[level]
    region = np.flatnonzero(largest_free_component(free).ravel())
    points = [cell_centre(rng.choice(region))]
    total = 0.0
    while total < length_m:
        for _ in range(max_tries):
            cand = cell_centre(rng.choice(region))
            leg = float(np.linalg.norm(cand - points[-1]))
            if leg > 0.0 and segment_is_free(free, points[-1], cand):
                break
        else:
            raise RuntimeError("no reachable waypoint found")
        points.append(cand)
        total += leg
    return np.stack(points)


def along_path(waypoints: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    """Positions ``[n, 2]`` at arc lengths ``distance_m`` along a polyline."""
    legs = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(legs)])
    d = np.clip(np.asarray(distance_m, dtype=np.float64), 0.0, cum[-1])
    j = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(legs) - 1)
    frac = (d - cum[j]) / legs[j]
    return waypoints[j] + frac[:, None] * (waypoints[j + 1] - waypoints[j])


def target_tracks(
    los: AerialLos,
    seed_key: list[int],
    levels: tuple[int, ...],
    speed_m_s: float,
    n_slots: int,
    slot_s: float = 0.5,
    max_speed_m_s: float | None = None,
) -> np.ndarray:
    """Positions ``[n_slots, Q, 3]`` of ``Q = len(levels)`` UAVs, one per entry of ``levels``.

    Target ``q`` flies at flight level ``levels[q]`` (an index into ``los.levels_m``) on the
    random-waypoint path drawn with ``numpy.random.default_rng([*seed_key, q])``, at constant
    speed; slot ``t`` is the position after ``t * slot_s`` seconds.  The path is drawn long
    enough for ``max_speed_m_s`` (default: ``speed_m_s``), so that the tracks of one seed at
    different speeds follow the same path.
    """
    longest = (max_speed_m_s or speed_m_s) * slot_s * n_slots
    distance = speed_m_s * slot_s * np.arange(n_slots)
    out = np.empty((n_slots, len(levels), 3))
    for q, level in enumerate(levels):
        path = waypoint_path(los, level, np.random.default_rng([*seed_key, q]), longest)
        out[:, q, :2] = along_path(path, distance)
        out[:, q, 2] = los.levels_m[level]
    return out


# ------------------------------------------------------------------------------ physical scale
def sensing_scale(
    *,
    power_w: float = 1.0,
    rcs_m2: float = 0.1,
    bandwidth_hz: float = 100.0e6,
    noise_figure_db: float = 7.0,
    processing_gain_db: float = 44.0,
    carrier_hz: float = CARRIER_FREQUENCY_HZ,
) -> float:
    """Factor ``kappa`` (1/m^2) of ``J = kappa * sum g_m g_n (u_m + u_n)(u_m + u_n)^T``.

    A transmitter ``m`` and a receiver ``n`` measure the bistatic range
    ``|p - t_m| + |p - r_n|`` of a target at ``p``; its gradient with respect to ``p`` is
    ``u_m + u_n``.  The Cramer-Rao variance of the range is
    ``sigma_r^2 = c^2 / (8 pi^2 beta^2 SNR)`` with the RMS bandwidth ``beta = B / sqrt(12)``
    of a flat spectrum of width ``B`` and the radar SNR
    ``P sigma_rcs (4 pi / lambda^2) g_m g_n G_p / (N0 B)`` for isotropic antennas, where
    ``g = (lambda / (4 pi d))^2`` is the free-space gain of one leg.  Hence
    ``kappa = (8 pi^2 beta^2 / c^2) * P sigma_rcs 4 pi G_p / (lambda^2 N0 B)``.
    """
    wavelength = SPEED_OF_LIGHT / carrier_hz
    beta_sq = bandwidth_hz**2 / 12.0
    n0 = BOLTZMANN_J_PER_K * REFERENCE_TEMPERATURE_K * 10.0 ** (noise_figure_db / 10.0)
    gain = 10.0 ** (processing_gain_db / 10.0)
    snr_factor = power_w * rcs_m2 * 4.0 * math.pi * gain / (wavelength**2 * n0 * bandwidth_hz)
    return 8.0 * math.pi**2 * beta_sq / SPEED_OF_LIGHT**2 * snr_factor


def processing_gain_db_for_scale(scale: float = 1.0e20, **kwargs) -> float:
    """Processing gain (dB) for which :func:`sensing_scale` equals ``scale``."""
    kwargs = {k: v for k, v in kwargs.items() if k != "processing_gain_db"}
    return 10.0 * math.log10(scale / sensing_scale(processing_gain_db=0.0, **kwargs))


def fim_matrix(entries) -> np.ndarray:
    """Symmetric matrices ``[..., 3, 3]`` from the six entries (11, 22, 33, 12, 13, 23)."""
    j11, j22, j33, j12, j13, j23 = entries
    out = np.empty((*np.shape(j11), 3, 3))
    for (r, c), v in zip(
        ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)),
        (j11, j22, j33, j12, j13, j23),
        strict=True,
    ):
        out[..., r, c] = v
        out[..., c, r] = v
    return out


def position_error_bound(fim: np.ndarray, eps: float) -> np.ndarray:
    """Bayesian position error bound ``sqrt(trace((J + eps I)^-1))`` of matrices ``[..., 3, 3]``.

    With ``J`` in 1/m^2 (simulator scale ``1e20``) the bound is in metres; ``eps I`` is the
    information of a Gaussian prior with standard deviation ``1 / sqrt(eps)`` per axis, so
    the bound never exceeds ``sqrt(3 / eps)`` (173.2 m for ``eps = 1e-4``).
    """
    eig = np.maximum(np.linalg.eigvalsh(fim), 0.0)
    return np.sqrt((1.0 / (eig + eps)).sum(axis=-1))


def prior_error_bound(eps: float) -> float:
    """Position error bound without any information: ``sqrt(3 / eps)``."""
    return math.sqrt(3.0 / eps)
