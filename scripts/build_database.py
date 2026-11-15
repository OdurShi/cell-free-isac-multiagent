"""Derive the compact database shipped in ``data/`` from the original ray-traced arrays.

Input (the simulator's arrays in ``simulation/`` of this repository):

* ``paris_pg_database_3d.npy``  (11, 11, 7, 125, 125) float32, linear path gain
* ``candidate_positions_3d.npy``  (847, 3) float64

Output (``data/``):

* ``pathgain_db_u16.npz``: ``code`` (847, 125, 125) uint16, ``db_min``, ``db_step``,
  ``outdoor`` (125, 125) bool
* ``candidate_positions_3d.npy``: byte-identical copy of the input
* ``manifest.json``: shapes, SHA-256 of inputs and outputs, quantisation error

Usage:
    python scripts/build_database.py [--source DIR] [--out DIR]

``--source`` defaults to ``simulation/``, ``--out`` to ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from cfisac_ma import data
from cfisac_ma.runner import repo_root


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root() / "simulation",
        help="directory with the .npy files (default: simulation/)",
    )
    parser.add_argument("--out", type=Path, default=data.default_data_dir())
    args = parser.parse_args()

    src_db = args.source / "paris_pg_database_3d.npy"
    src_pos = args.source / "candidate_positions_3d.npy"
    original = np.load(src_db).reshape(-1, data.GRID_SIZE, data.GRID_SIZE)
    threshold = np.float32(data.blockage_threshold_linear())

    code, db_min, db_step = data.encode_path_gain(original)
    # Outdoor rule of the simulator, evaluated on the original (unthresholded) gains.
    outdoor = original.mean(axis=0) > threshold

    args.out.mkdir(parents=True, exist_ok=True)
    out_db = args.out / data.DATABASE_FILE
    np.savez_compressed(
        out_db, code=code, db_min=np.float64(db_min), db_step=np.float64(db_step), outdoor=outdoor
    )
    out_pos = args.out / data.POSITIONS_FILE
    shutil.copyfile(src_pos, out_pos)

    # Error of the shipped file against the original.
    decoded = data.decode_path_gain(code, db_min, db_step)
    blocked = original < threshold
    assert np.array_equal(decoded == 0.0, blocked), "blocked mask changed"
    orig_db = 10.0 * np.log10(original[~blocked].astype(np.float64))
    dec_db = 10.0 * np.log10(decoded[~blocked])
    rel = np.abs(decoded[~blocked] / original[~blocked].astype(np.float64) - 1.0)
    valid_original = blocked.mean(axis=(1, 2)) <= data.MAX_BLOCKED_FRACTION
    valid_shipped = data.valid_site_mask(code)
    assert np.array_equal(valid_original, valid_shipped), "valid-site set changed"
    outdoor_from_shipped = decoded.mean(axis=0) > threshold

    manifest = {
        "source": {
            "paris_pg_database_3d.npy": {
                "sha256": sha256(src_db),
                "shape": [11, 11, 7, 125, 125],
                "dtype": str(np.load(src_db, mmap_mode="r").dtype),
                "bytes": src_db.stat().st_size,
            },
            "candidate_positions_3d.npy": {"sha256": sha256(src_pos), "shape": [847, 3]},
        },
        "shipped": {
            data.DATABASE_FILE: {"sha256": sha256(out_db), "bytes": out_db.stat().st_size},
            data.POSITIONS_FILE: {"sha256": sha256(out_pos), "bytes": out_pos.stat().st_size},
        },
        "encoding": {
            "blocked_code": 0,
            "blockage_threshold_db": data.BLOCKAGE_THRESHOLD_DB,
            "db_min": db_min,
            "db_step": db_step,
            "db_max": db_min + (data.N_LEVELS - 1) * db_step,
            "levels": data.N_LEVELS,
        },
        "statistics": {
            "candidates": int(len(code)),
            "valid_candidates": int(valid_shipped.sum()),
            "blocked_fraction_all_entries": float(blocked.mean()),
            "outdoor_cells": int(outdoor.sum()),
        },
        "error_vs_original": {
            "blocked_mask_identical": True,
            "valid_site_set_identical": True,
            "max_abs_error_db": float(np.abs(dec_db - orig_db).max()),
            "max_relative_error_linear": float(rel.max()),
            "rms_error_db": float(np.sqrt(np.mean((dec_db - orig_db) ** 2))),
            "outdoor_cells_differing_if_recomputed_from_shipped_gains": int(
                (outdoor != outdoor_from_shipped).sum()
            ),
        },
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest["error_vs_original"], indent=2))
    print(f"wrote {out_db} ({out_db.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
