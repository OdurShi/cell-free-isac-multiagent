"""Ray-trace the scene with Sionna RT: line-of-sight tables for aerial targets, database check.

This is the only script of the placement package that needs Sionna RT (``pip install
sionna-rt``, or ``pip install -e ".[rt]"``).  Everything else runs on the files it writes.

Three jobs:

``--los`` (default)
    Exact line-of-sight tests on the built-in ``etoile`` scene, the scene the simulator's
    path-gain database was traced on.  For every valid candidate site and every cell of the
    125 x 125 grid (4 m cells), a ray from the site to the cell centre is tested against the
    scene geometry at the flight levels 10, 20, 30 and 40 m and at the user height of 1.5 m.
    Writes ``data/aerial_los.npz`` and ``data/aerial_manifest.json``.

``--check-database``
    Traces the radio map of every candidate position again with the settings of
    ``simulation/generate_offline_database.py`` and compares it with the stored database
    (``simulation/paris_pg_database_3d.npy``).  Writes ``results/e8_database_check.json``.

``--check-lookup``
    Compares the lookup the package uses (``cfisac_ma.aerial.AerialLos.visible``: the table
    entry of the cell that contains the target) with exact tests on the scene, to the centre
    of that cell and to a random position inside it.  Writes
    ``results/e8_los_lookup_check.json``.

Usage:
    python scripts/raytrace_scene.py [--los] [--check-database] [--check-lookup] [--figure]
                                     [--data-out DIR] [--results-out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from cfisac_ma import data
from cfisac_ma.runner import repo_root

FLIGHT_LEVELS_M = (10.0, 20.0, 30.0, 40.0)
USER_HEIGHT_M = 1.5
#: A cell is free at a flight level if the highest roof inside it is this far below the level.
CLEARANCE_M = 2.0
LOS_FILE = "aerial_los.npz"
LOS_MANIFEST = "aerial_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_etoile():
    import sionna.rt as rt

    return rt.load_scene(rt.scene.etoile)


def cell_centres() -> np.ndarray:
    """``[125 * 125, 2]`` x, y of the cell centres, row-major with row = y and column = x."""
    c = data.AREA_MIN_M + (np.arange(data.GRID_SIZE) + 0.5) * data.CELL_SIZE_M
    gx, gy = np.meshgrid(c, c)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def line_of_sight(mi_scene, origin: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``[n]`` booleans: the segment from ``origin`` ``[3]`` to each of ``points`` is free."""
    import mitsuba as mi

    d = points - origin[None, :]
    dist = np.linalg.norm(d, axis=1)
    d = d / np.maximum(dist, 1e-9)[:, None]
    o = np.broadcast_to(origin, points.shape).astype(np.float32)
    d32 = d.astype(np.float32)
    ray = mi.Ray3f(
        mi.Point3f(*(mi.Float(np.ascontiguousarray(o[:, i])) for i in range(3))),
        mi.Vector3f(*(mi.Float(np.ascontiguousarray(d32[:, i])) for i in range(3))),
    )
    ray.maxt = mi.Float((dist * (1.0 - 1e-4)).astype(np.float32))
    return ~np.array(mi_scene.ray_test(ray), dtype=bool)


def roof_height(mi_scene, xy: np.ndarray) -> np.ndarray:
    """Height of the highest surface above each ground position ``[n, 2]`` (0 = ground)."""
    import mitsuba as mi

    top = 500.0
    o = np.concatenate([xy, np.full((len(xy), 1), top)], axis=1).astype(np.float32)
    ray = mi.Ray3f(
        mi.Point3f(*(mi.Float(np.ascontiguousarray(o[:, i])) for i in range(3))),
        mi.Vector3f(0, 0, -1),
    )
    si = mi_scene.ray_intersect(ray)
    t = np.array(si.t, dtype=np.float64)
    hit = np.array(si.is_valid(), dtype=bool)
    return np.where(hit, top - t, 0.0)


def versions() -> dict:
    import drjit
    import mitsuba as mi
    import sionna.rt as rt

    return {
        "sionna_rt": getattr(rt, "__version__", "unknown"),
        "mitsuba": mi.__version__,
        "drjit": drjit.__version__,
        "variant": mi.variant(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def build_los(out: Path, figure: Path | None) -> None:
    scene_data = data.load_scene()
    scene = load_etoile()
    mi_scene = scene.mi_scene
    centres = cell_centres()
    n_cells = len(centres)
    sites = scene_data.positions  # [345, 3]

    t0 = time.time()
    levels = np.array(FLIGHT_LEVELS_M)
    los = np.zeros((len(levels), len(sites), n_cells), dtype=bool)
    ground = np.zeros((len(sites), n_cells), dtype=bool)
    for j, site in enumerate(sites):
        for li, z in enumerate(levels):
            pts = np.concatenate([centres, np.full((n_cells, 1), z)], axis=1)
            los[li, j] = line_of_sight(mi_scene, site, pts)
        pts = np.concatenate([centres, np.full((n_cells, 1), USER_HEIGHT_M)], axis=1)
        ground[j] = line_of_sight(mi_scene, site, pts)

    # Highest roof inside each 4 m cell, from a 1 m grid of downward rays.
    sub = data.AREA_MIN_M + (np.arange(4 * data.GRID_SIZE) + 0.5)
    sx, sy = np.meshgrid(sub, sub)
    roof_1m = roof_height(mi_scene, np.stack([sx.ravel(), sy.ravel()], axis=1))
    roof_1m = roof_1m.reshape(4 * data.GRID_SIZE, 4 * data.GRID_SIZE)
    roof_cell = roof_1m.reshape(data.GRID_SIZE, 4, data.GRID_SIZE, 4).max(axis=(1, 3))
    free = np.stack([roof_cell < z - CLEARANCE_M for z in levels])  # [L, 125, 125]
    site_roof = roof_height(mi_scene, sites[:, :2])
    elapsed = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    path = out / LOS_FILE
    np.savez_compressed(
        path,
        levels_m=levels,
        los_bits=np.packbits(los, axis=-1),
        ground_los_bits=np.packbits(ground, axis=-1),
        free=free,
        site_ids=scene_data.site_ids.astype(np.int64),
        site_roof_m=np.round(site_roof, 2),
    )

    unblocked = (scene_data.gain > 0.0).reshape(len(sites), -1)
    free_flat = free.reshape(len(levels), -1)
    manifest = {
        "file": {LOS_FILE: {"sha256": sha256(path), "bytes": path.stat().st_size}},
        "source": {
            "scene": "sionna.rt.scene.etoile",
            "path_gain_database_sha256": sha256(data.default_data_dir() / data.DATABASE_FILE),
        },
        "software": versions(),
        "content": {
            "levels_m": list(FLIGHT_LEVELS_M),
            "user_height_m": USER_HEIGHT_M,
            "clearance_m": CLEARANCE_M,
            "sites": int(len(sites)),
            "cells": int(n_cells),
            "los_bits": "uint8 [levels, sites, 1954]: numpy.packbits of [levels, sites, 15625]",
            "ground_los_bits": "uint8 [sites, 1954]: numpy.packbits of [sites, 15625]",
            "free": "bool [levels, 125, 125]: highest roof in the cell below level - clearance",
        },
        "statistics": {
            "los_fraction_all_cells": {
                f"{z:g}": float(los[i].mean()) for i, z in enumerate(levels)
            },
            "los_fraction_free_cells": {
                f"{z:g}": float(los[i][:, free_flat[i]].mean()) for i, z in enumerate(levels)
            },
            "free_cell_fraction": {f"{z:g}": float(free[i].mean()) for i, z in enumerate(levels)},
            "ground_los_fraction": float(ground.mean()),
            "database_unblocked_fraction": float(unblocked.mean()),
            "database_unblocked_given_ground_los": float(unblocked[ground].mean()),
            "ground_los_given_database_unblocked": float(ground[unblocked].mean()),
            "built_area_fraction_1m_grid": float((roof_1m > 2.0).mean()),
            "roof_height_quantiles_m": {
                q: float(np.quantile(roof_1m[roof_1m > 2.0], float(q)))
                for q in ("0.1", "0.5", "0.9", "0.99")
            },
            "max_roof_m": float(roof_1m.max()),
            "valid_sites_below_a_roof": int((site_roof > sites[:, 2]).sum()),
        },
        "seconds": round(elapsed, 1),
    }
    with open(out / LOS_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest["statistics"], indent=2))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB) in {elapsed:.1f} s")

    if figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
        extent = (data.AREA_MIN_M, data.AREA_MAX_M, data.AREA_MIN_M, data.AREA_MAX_M)
        im = ax.imshow(roof_1m, origin="lower", extent=extent, cmap="cividis", vmin=0, vmax=50)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="Height of the highest surface (m)", shrink=0.85)
        fig.tight_layout()
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure)
        print(f"wrote {figure}")


def check_database(out: Path, depth_sample: int) -> None:
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter

    source = repo_root() / "simulation"
    db = np.load(source / "paris_pg_database_3d.npy", mmap_mode="r")
    db = db.reshape(-1, data.GRID_SIZE, data.GRID_SIZE)
    positions = np.load(source / "candidate_positions_3d.npy")
    threshold = np.float32(data.blockage_threshold_linear())

    scene = load_etoile()
    scene.frequency = 28e9
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    solver = RadioMapSolver()

    def trace(index: int, max_depth: int) -> np.ndarray:
        scene.add(
            Transmitter(
                name="tx", position=[float(v) for v in positions[index]], orientation=[0, 0, 0]
            )
        )
        radio_map = solver(
            scene,
            max_depth=max_depth,
            cell_size=[4, 4],
            center=[0, 0, 1.5],
            size=[500, 500],
            orientation=[0, 0, 0],
            samples_per_tx=int(5e6),
        )
        gain = np.squeeze(radio_map.path_gain.numpy()).astype(np.float32)
        scene.remove("tx")
        return gain

    def compare(new: np.ndarray, ref: np.ndarray) -> dict:
        a, b = new >= threshold, ref >= threshold
        both = a & b
        row = {"mask_agreement": float((a == b).mean()), "common_cells": int(both.sum())}
        if both.any():
            err = np.abs(10 * np.log10(new[both].astype(np.float64)) - 10 * np.log10(ref[both]))
            row.update(max_abs_error_db=float(err.max()), mean_abs_error_db=float(err.mean()))
        return row

    t0 = time.time()
    rows = []
    for index in range(len(positions)):
        row = {"site": index, **compare(trace(index, 3), np.asarray(db[index], dtype=np.float64))}
        rows.append(row)
    t_all = time.time() - t0
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(len(positions), size=depth_sample, replace=False).tolist())
    rows4 = [
        {"site": i, **compare(trace(i, 4), np.asarray(db[i], dtype=np.float64))} for i in sample
    ]

    def summary(rs: list[dict]) -> dict:
        err = [r["max_abs_error_db"] for r in rs if "max_abs_error_db" in r]
        mean = [r["mean_abs_error_db"] for r in rs if "mean_abs_error_db" in r]
        return {
            "sites": len(rs),
            "sites_with_identical_mask": int(sum(r["mask_agreement"] == 1.0 for r in rs)),
            "min_mask_agreement": float(min(r["mask_agreement"] for r in rs)),
            "mean_mask_agreement": float(np.mean([r["mask_agreement"] for r in rs])),
            "max_abs_error_db": float(max(err)) if err else None,
            "mean_abs_error_db": float(np.mean(mean)) if mean else None,
        }

    result = {
        "settings": {
            "frequency_hz": 28e9,
            "cell_size_m": [4, 4],
            "plane_height_m": 1.5,
            "samples_per_tx": int(5e6),
            "antenna": "isotropic, vertical polarisation",
            "blockage_threshold_db": data.BLOCKAGE_THRESHOLD_DB,
        },
        "software": versions(),
        "max_depth_3": summary(rows),
        "max_depth_4_sample": summary(rows4),
        "seconds_all_sites_depth_3": round(t_all, 1),
        "per_site_depth_3": rows,
        "per_site_depth_4_sample": rows4,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e8_database_check.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("max_depth_3", "max_depth_4_sample")}, indent=2))
    print(f"wrote {path} ({t_all:.1f} s for {len(positions)} radio maps)")


def check_lookup(out: Path, cells_per_level: int = 400, sites_per_level: int = 40) -> None:
    """Table lookup of the package against exact visibility tests (seeded sample)."""
    from cfisac_ma.aerial import cell_centre, load_aerial_los

    los = load_aerial_los()
    sites = data.load_scene().positions
    mi_scene = load_etoile().mi_scene
    rng = np.random.default_rng(12345)
    half = 0.5 * data.CELL_SIZE_M - 1e-3
    rows = []
    for level, z in enumerate(los.levels_m):
        cells = rng.choice(np.flatnonzero(los.free[level].ravel()), cells_per_level, replace=False)
        centre = cell_centre(cells)
        inside = centre + rng.uniform(-half, half, size=centre.shape)
        height = np.full(len(cells), z)
        at_centre = np.column_stack([centre, height])
        at_point = np.column_stack([inside, height])
        if not (los.cell_of(at_point) == cells).all():
            raise RuntimeError("a sampled position left its cell")
        table = los.visible(at_point)
        pairs = same_centre = same_point = 0
        for site in rng.choice(len(sites), sites_per_level, replace=False):
            origin = sites[site].astype(np.float64)
            pairs += len(cells)
            same_centre += int((line_of_sight(mi_scene, origin, at_centre) == table[site]).sum())
            same_point += int((line_of_sight(mi_scene, origin, at_point) == table[site]).sum())
        rows.append(
            {
                "level_m": float(z),
                "pairs": pairs,
                "agreement_cell_centre": same_centre / pairs,
                "agreement_position_in_cell": same_point / pairs,
            }
        )
    total = sum(r["pairs"] for r in rows)

    def pooled(key: str) -> float:
        return sum(r[key] * r["pairs"] for r in rows) / total

    result = {
        "sample": {
            "seed": 12345,
            "free_cells_per_level": cells_per_level,
            "sites_per_level": sites_per_level,
        },
        "software": versions(),
        "pairs": total,
        "agreement_cell_centre": pooled("agreement_cell_centre"),
        "agreement_position_in_cell": pooled("agreement_position_in_cell"),
        "per_level": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e8_los_lookup_check.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "software"}, indent=2))
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--los", action="store_true", help="write the line-of-sight tables")
    parser.add_argument("--check-database", action="store_true")
    parser.add_argument("--check-lookup", action="store_true")
    parser.add_argument("--data-out", type=Path, default=data.default_data_dir())
    parser.add_argument("--results-out", type=Path, default=repo_root() / "results")
    parser.add_argument("--figure", action="store_true", help="also draw the roof-height map")
    parser.add_argument("--depth-sample", type=int, default=60)
    args = parser.parse_args()
    if not (args.los or args.check_database or args.check_lookup):
        args.los = True
    if args.los:
        figure = repo_root() / "docs" / "figures" / "etoile_roof_height.png"
        build_los(args.data_out, figure if args.figure else None)
    if args.check_database:
        check_database(args.results_out, args.depth_sample)
    if args.check_lookup:
        check_lookup(args.results_out)


if __name__ == "__main__":
    main()
