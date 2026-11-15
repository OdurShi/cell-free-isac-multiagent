"""Compact path-gain database: encoding, decoding and loading.

The ray-traced database of the 3-D scene stores, for each of 847 candidate access-point
(AP) positions, the linear path gain to a 125 x 125 ground grid (4 m cells over
[-250, 250] m, receiver height 1.5 m).  The repository ships a compact version of it:

* gains below the blockage threshold of -110 dB are stored as code 0 ("blocked");
* all other gains are stored in dB, quantised uniformly to the codes 1 ... 65535 over
  [-110 dB, ``db_max``].

``scripts/build_database.py`` creates the file from the original array with
:func:`encode_path_gain`; :func:`load_scene` reads it back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Gains below this value are treated as blocked (set to zero), as in the simulator.
BLOCKAGE_THRESHOLD_DB = -110.0
#: A candidate position is valid if at most this fraction of its map is blocked.
MAX_BLOCKED_FRACTION = 0.80
#: Ground grid of the database.
GRID_SIZE = 125
CELL_SIZE_M = 4.0
AREA_MIN_M = -250.0
AREA_MAX_M = 250.0

DATABASE_FILE = "pathgain_db_u16.npz"
POSITIONS_FILE = "candidate_positions_3d.npy"
N_LEVELS = 65535  # codes 1 ... 65535


def default_data_dir() -> Path:
    """Directory of the shipped data: ``$CFISAC_MA_DATA`` or ``<repository>/data``."""
    env = os.environ.get("CFISAC_MA_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def blockage_threshold_linear() -> float:
    return 10.0 ** (BLOCKAGE_THRESHOLD_DB / 10.0)


def encode_path_gain(gain: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantise linear path gains to ``uint16`` codes.

    Args:
        gain: linear path gains, any shape, non-negative.

    Returns:
        ``(code, db_min, db_step)``.  ``code == 0`` marks a blocked entry (gain below
        -110 dB, compared exactly as the simulator does: ``gain < float32(1e-11)``).  For
        other entries the gain in dB is ``db_min + (code - 1) * db_step``.
    """
    gain = np.asarray(gain)
    blocked = gain < np.float32(blockage_threshold_linear())
    gain_db = 10.0 * np.log10(gain[~blocked].astype(np.float64))
    db_min = BLOCKAGE_THRESHOLD_DB
    db_step = (float(gain_db.max()) - db_min) / (N_LEVELS - 1)
    level = np.clip(np.rint((gain_db - db_min) / db_step), 0, N_LEVELS - 1)
    code = np.zeros(gain.shape, dtype=np.uint16)
    code[~blocked] = (level + 1).astype(np.uint16)
    return code, db_min, db_step


def decode_path_gain(code: np.ndarray, db_min: float, db_step: float) -> np.ndarray:
    """Inverse of :func:`encode_path_gain`: linear gains (float64), blocked entries are 0."""
    code = np.asarray(code)
    gain_db = db_min + (code.astype(np.float64) - 1.0) * db_step
    return np.where(code > 0, 10.0 ** (gain_db / 10.0), 0.0)


@dataclass(frozen=True)
class Scene:
    """The ray-traced scene restricted to the valid candidate sites.

    Attributes:
        positions: ``[V, 3]`` coordinates (m) of the valid candidate sites.
        site_ids: ``[V]`` index of each valid site in the full list of 847 candidates.
        gain: ``[V, 125, 125]`` linear path gain to the ground grid (row = y, column = x);
            blocked entries are exactly 0.
        outdoor: ``[125, 125]`` boolean mask of the cells the simulator treats as outdoor
            (mean gain over all 847 candidate positions above -110 dB).
        all_positions: ``[847, 3]`` coordinates of all candidate positions.
    """

    positions: np.ndarray
    site_ids: np.ndarray
    gain: np.ndarray
    outdoor: np.ndarray
    all_positions: np.ndarray

    @property
    def n_sites(self) -> int:
        return len(self.site_ids)

    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid cell ``(row, col)`` of ground positions ``[..., 2]`` (simulator convention)."""
        return cell_index(xy)

    def nearest_sites(self, k: int) -> np.ndarray:
        """``[V, k]`` indices of the ``k`` nearest other valid sites of every valid site."""
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(dist, np.inf)
        return np.argsort(dist, axis=1, kind="stable")[:, :k]


def cell_index(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell ``(row, col)`` of ground positions, ``row`` from y and ``col`` from x."""
    xy = np.asarray(xy, dtype=np.float64)
    col = np.clip(((xy[..., 0] - AREA_MIN_M) / CELL_SIZE_M).astype(np.int64), 0, GRID_SIZE - 1)
    row = np.clip(((xy[..., 1] - AREA_MIN_M) / CELL_SIZE_M).astype(np.int64), 0, GRID_SIZE - 1)
    return row, col


def valid_site_mask(code_or_gain: np.ndarray) -> np.ndarray:
    """Valid candidates: at most 80% of the map blocked.

    Accepts the ``uint16`` codes (blocked = 0) or thresholded linear gains (blocked = 0).
    """
    blocked_fraction = (np.asarray(code_or_gain) == 0).mean(axis=(1, 2))
    return blocked_fraction <= MAX_BLOCKED_FRACTION


def load_scene(data_dir: str | os.PathLike | None = None) -> Scene:
    """Load the shipped compact database and keep the valid candidate sites."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    with np.load(data_dir / DATABASE_FILE) as f:
        code = f["code"]
        db_min = float(f["db_min"])
        db_step = float(f["db_step"])
        outdoor = f["outdoor"].astype(bool)
    all_positions = np.load(data_dir / POSITIONS_FILE)
    code = code.reshape(-1, GRID_SIZE, GRID_SIZE)
    if len(code) != len(all_positions):
        raise ValueError("database and candidate positions disagree in length")
    site_ids = np.flatnonzero(valid_site_mask(code))
    gain = decode_path_gain(code[site_ids], db_min, db_step)
    return Scene(
        positions=all_positions[site_ids].astype(np.float64),
        site_ids=site_ids,
        gain=gain,
        outdoor=outdoor,
        all_positions=all_positions.astype(np.float64),
    )
