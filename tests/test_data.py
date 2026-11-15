"""The shipped database: encoding, decoding, valid sites, manifest."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cfisac_ma import data
from cfisac_ma.runner import repo_root


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(0)
    gain = 10.0 ** (rng.uniform(-125.0, -70.0, size=(4, 20, 20)) / 10.0)
    gain[0, :5] = 0.0
    code, db_min, db_step = data.encode_path_gain(gain.astype(np.float32))
    decoded = data.decode_path_gain(code, db_min, db_step)
    blocked = gain.astype(np.float32) < np.float32(1e-11)
    assert np.array_equal(decoded == 0.0, blocked)
    err_db = np.abs(10 * np.log10(decoded[~blocked]) - 10 * np.log10(gain[~blocked]))
    assert err_db.max() <= db_step / 2 + 1e-5  # half a quantisation step (+ float32 input rounding)
    assert code.dtype == np.uint16 and code[~blocked].min() >= 1


def test_scene_shapes_and_valid_sites(scene):
    assert scene.gain.shape == (scene.n_sites, 125, 125)
    assert scene.n_sites == 345
    assert scene.all_positions.shape == (847, 3)
    assert scene.outdoor.shape == (125, 125) and scene.outdoor.dtype == bool
    blocked_fraction = (scene.gain == 0.0).mean(axis=(1, 2))
    assert blocked_fraction.max() <= data.MAX_BLOCKED_FRACTION
    positive = scene.gain[scene.gain > 0]
    assert positive.min() >= 10 ** (-110.0 / 10) * (1 - 1e-6)
    assert np.allclose(scene.positions, scene.all_positions[scene.site_ids])
    assert set(np.unique(scene.positions[:, 2])) <= {5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0}


def test_manifest_matches_shipped_files():
    root = data.default_data_dir()
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["shipped"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
    assert manifest["statistics"]["valid_candidates"] == 345
    assert manifest["error_vs_original"]["max_abs_error_db"] < 1e-3


def test_cell_index_convention():
    row, col = data.cell_index(np.array([[-250.0, -250.0], [249.9, -250.0], [0.0, 100.0]]))
    assert row.tolist() == [0, 0, 87] and col.tolist() == [0, 124, 62]
    # positions outside the area are clipped to the border cells, as in the simulator
    row, col = data.cell_index(np.array([[400.0, -400.0]]))
    assert (row[0], col[0]) == (0, 124)


def test_compact_database_matches_the_simulator_arrays(scene):
    """``data/`` is the simulator's database (``simulation/``), thresholded and quantised."""
    source = repo_root() / "simulation"
    if not (source / "paris_pg_database_3d.npy").exists():
        pytest.skip("the simulator's arrays are not present")
    manifest = json.loads((data.default_data_dir() / "manifest.json").read_text())
    for name, entry in manifest["source"].items():
        digest = hashlib.sha256((source / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name

    positions = np.load(source / "candidate_positions_3d.npy")
    assert np.array_equal(positions, scene.all_positions)

    original = np.load(source / "paris_pg_database_3d.npy", mmap_mode="r")
    original = np.asarray(original).reshape(-1, data.GRID_SIZE, data.GRID_SIZE)
    blocked = original < np.float32(data.blockage_threshold_linear())
    valid = np.flatnonzero(blocked.mean(axis=(1, 2)) <= data.MAX_BLOCKED_FRACTION)
    assert np.array_equal(valid, scene.site_ids)

    kept, kept_blocked = original[valid].astype(np.float64), blocked[valid]
    assert np.array_equal(scene.gain == 0.0, kept_blocked)
    err_db = np.abs(10 * np.log10(scene.gain[~kept_blocked]) - 10 * np.log10(kept[~kept_blocked]))
    assert err_db.max() <= manifest["encoding"]["db_step"] / 2 + 1e-9
    assert np.array_equal(scene.outdoor, original.mean(axis=0) > np.float32(1e-11))
